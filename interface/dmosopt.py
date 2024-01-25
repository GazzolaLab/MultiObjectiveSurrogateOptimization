from machinable import Component
from mpi4py import MPI
from pydantic import BaseModel, ConfigDict, Field, field_validator, TypeAdapter
from dmosopt import dmosopt
from dmosopt import config
from machinable.config import to_dict
from typing import Dict, Optional, List, Callable, Literal, Set, Any, Union, Tuple
import copy
from machinable.config import match_method
import os
import sys
import inspect
import h5py
from dmosopt.dmosopt import init_from_h5
from dmosopt.MOASMO import get_best
from dmosopt.hv import HyperVolume
from dmosopt import indicators
import matplotlib.pyplot as plt
import numpy as np

sys_excepthook = sys.excepthook


def mpi_excepthook(type, value, traceback):
    sys_excepthook(type, value, traceback)
    sys.stdout.flush()
    sys.stderr.flush()
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)


sys.excepthook = mpi_excepthook


class Dmosopt(Component):
    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid")

        dopt_params: Dict = Field("???")
        time_limit: Optional[int] = None
        feasible: bool = True
        return_features: bool = False
        return_constraints: bool = False
        spawn_workers: bool = False
        sequential_spawn: bool = False
        spawn_startup_wait: Optional[int] = None
        spawn_executable: Optional[str] = None
        spawn_args: List[str] = []
        nprocs_per_worker: int = 1
        collective_mode: Literal["gather", "sendrecv"] = "gather"
        verbose: bool = True
        worker_debug: bool = False

        @field_validator("dopt_params", mode="before")
        @classmethod
        def valid_optimization_settings(cls, params: Dict) -> Dict:
            _t = {
                "opt_id": str,
                "obj_fun_name": Optional[str],
                "obj_fun_init_name": Optional[str],
                "obj_fun_init_args": Dict,
                "controller_init_fun_name": Optional[str],
                "controller_init_fun_args": Dict,
                "reduce_fun_name": Optional[str],
                "reduce_fun_args": Dict,
                "broker_fun_name": Optional[str],
                "broker_module_name": Optional[str],
                # DistOptimizer
                "objective_names": List[str],
                "feature_dtypes": str,
                "constraint_names": List[str],
                "n_initial": int,
                "initial_maxiter": int,
                "initial_method": Union[
                    Callable,
                    Literal["glp", "slh", "lh", "mc", "sobol"],
                    Dict[str, Any],
                    str,
                ],
                "verbose": bool,
                "problem_ids": Optional[Set],
                "problem_parameters": Optional[Dict],
                "space": Optional[
                    Dict[str, Tuple[Union[int, float], Union[int, float]]]
                ],
                "population_size": int,
                "num_generations": int,
                "resample_fraction": float,
                "distance_metric": Union[Callable, Literal["crowding", "euclidean"]],
                "n_epochs": int,
                "save_eval": bool,
                "file_path": Optional[str],
                "save": bool,
                "save_surrogate_evals": bool,
                "save_optimizer_params": bool,
                "metadata": Any,
                "surrogate_method_name": Union[
                    str,
                    Literal[
                        "gpr",
                        "egp",
                        "megp",
                        "mdgp",
                        "mdspp",
                        "vgp",
                        "svgp",
                        "spv",
                        "siv",
                        "crv",
                    ],
                    None,
                ],
                "surrogate_method_kwargs": Dict,
                "surrogate_custom_training": Optional[str],
                "optimizer_name": Literal["nsga2", "age", "smpso", "cmaes"],
                "optimizer_kwargs": Dict,
                "sensitivity_method_name": Literal["dgsm", "fast"],
                "sensitivity_method_kwargs": Dict,
                "local_random": Any,
                "random_seed": Optional[int],
                "feasibility_model": bool,
                "termination_conditions": Optional[Dict],
                #
                "di_crossover": Any,  #
                "di_mutation": Any,  #
            }

            payload = copy.deepcopy(to_dict(params))
            for k, v in payload.items():
                if k not in _t:
                    raise ValueError(f"Invalid option: {k}")
                if isinstance(v, str) and match_method(v):
                    # config method allowed
                    continue
                try:
                    TypeAdapter(_t[k]).validate_python(v)
                except Exception as _ex:
                    print(v)
                    raise ValueError(
                        f"Invalid type for '{k}'; expected {_t[k]} but got:"
                    ) from _ex

            # additional rules
            if (payload.get("random_seed", None) is not None) and (
                payload.get("local_random", None) is not None
            ):
                raise ValueError(
                    "Both random_seed and local_random are specified! Only one or the other must be specified."
                )

            # validate imports eagerly
            for path, alias, kw in [
                ("obj_fun_name", {}, None),
                ("obj_fun_init_name", {}, "controller_init_fun_args"),
                ("controller_init_fun_name", {}, "controller_init_fun_args"),
                ("reduce_fun_name", {}, "reduce_fun_args"),
                ("broker_fun_name", {}, None),
                ("initial_method", config.default_sampling_methods, None),
                (
                    "surrogate_method_name",
                    config.default_surrogate_methods,
                    "surrogate_method_kwargs",
                ),
                ("surrogate_custom_training", {}, None),
                ("optimizer_name", config.default_optimizers, "optimizer_kwargs"),
                (
                    "sensitivity_method_name",
                    config.default_sa_methods,
                    "sensitivity_method_kwargs",
                ),
                ("feature_dtypes", {}, None),
            ]:
                if isinstance(target := payload.get(path, None), str):
                    if target in alias:
                        target = alias[target]
                    try:
                        obj = config.import_object_by_path(target)
                    except ImportError as _ex:
                        raise ValueError(
                            f"Could not resolve import path '{target}' for '{path}'"
                        ) from _ex

                    if (d := payload.get(kw, None)) is not None:
                        # verify arguments
                        sig = inspect.signature(obj)
                        for key in d.keys():
                            if key not in sig.parameters:
                                message = ""
                                for name, param in sig.parameters.items():
                                    if param.default is param.empty:
                                        message += f"{name}, "
                                    else:
                                        message += f"{name}={param.default}, "
                                raise ValueError(
                                    f"Invalid {kw} for {target}. Found `{key}`, but signature is {message[:-2]}"
                                )

            return payload

    def __call__(self) -> None:
        params = to_dict(self.config.dopt_params)
        if "file_path" not in params:
            params["file_path"] = self.output_filepath
        if "local_random" not in params and "random_seed" not in params:
            params["random_seed"] = self.seed
        if "feature_dtypes" in params:
            feature_dtypes = config.import_object_by_path(params["feature_dtypes"])
            if callable(feature_dtypes):
                feature_dtypes = feature_dtypes(self)
            params["feature_dtypes"] = feature_dtypes
        dmosopt.run(
            dopt_params=params,
            time_limit=self.config.time_limit,
            feasible=self.config.feasible,
            return_features=self.config.return_features,
            return_constraints=self.config.return_constraints,
            spawn_workers=self.config.spawn_workers,
            sequential_spawn=self.config.sequential_spawn,
            spawn_startup_wait=self.config.spawn_startup_wait,
            spawn_executable=self.config.spawn_executable,
            spawn_args=self.config.spawn_args,
            nprocs_per_worker=self.config.nprocs_per_worker,
            collective_mode=self.config.collective_mode,
            verbose=self.config.verbose,
            worker_debug=self.config.worker_debug,
        )

    def config_use(self, path):
        return config.import_object_by_path(path)

    @property
    def output_filepath(self) -> str:
        return os.path.abspath(self.local_directory("dmosopt.h5"))

    def on_write_meta_data(self):
        return MPI.COMM_WORLD.Get_rank() == 0

    def results(self):
        opt_id = self.config.dopt_params.opt_id
        (
            _,
            max_epoch,
            old_evals,
            param_names,
            is_int,
            lo_bounds,
            hi_bounds,
            objective_names,
            feature_names,
            constraint_names,
            problem_parameters,
            problem_ids,
        ) = init_from_h5(self.output_filepath, None, opt_id, None)

        problem_id = 0

        with h5py.File(self.output_filepath, "r") as f:
            # metadata = f[f'/{self.config.optimizer.opt_id}/metadata'][:]
            predictions = f[f"{opt_id}/{problem_id}/predictions"][:]
            objectives = f[f"{opt_id}/{problem_id}/objectives"][:]
            epochs = f[f"/{opt_id}/{problem_id}/epochs"][:]

        old_eval_epochs = [e.epoch for e in old_evals[problem_id]]
        old_eval_xs = [e.parameters for e in old_evals[problem_id]]
        old_eval_ys = [e.objectives for e in old_evals[problem_id]]
        x = np.vstack(old_eval_xs)
        y = np.vstack(old_eval_ys)
        old_eval_fs = None
        f = None
        if feature_names is not None:
            old_eval_fs = [e.features for e in old_evals[problem_id]]
            f = np.concatenate(old_eval_fs, axis=None)

        old_eval_cs = None
        c = None
        if constraint_names is not None:
            old_eval_cs = [e.constraints for e in old_evals[problem_id]]
            c = np.vstack(old_eval_cs)

        x = np.vstack(old_eval_xs)
        y = np.vstack(old_eval_ys)

        if len(old_eval_epochs) > 0 and old_eval_epochs[0] is not None:
            epochs = np.concatenate(old_eval_epochs, axis=None)

        n_dim = len(lo_bounds)
        n_objectives = len(objective_names)

        predictions_array = np.column_stack(
            tuple(predictions[x] for x in predictions.dtype.names)
        )
        objectives_array = np.column_stack(
            tuple(objectives[x] for x in objectives.dtype.names)
        )

        best_x, best_y, best_f, best_c, best_epoch, _ = get_best(
            x,
            y,
            f,
            c,
            len(param_names),
            len(objective_names),
            epochs=epochs,
            feasible=True,
        )

        return locals()

    def pareto_front(self):
        results = self.results()
        return np.stack((results["best_x"][:, 0], results["best_y"][:, 1])).T

    def hypervolume(self, reference):
        pf = self.pareto_front()
        indicator = indicators.Hypervolume(ref_point=reference, pf=pf)
        return indicator.do(pf)

    def igd(self, reference):
        pf = self.pareto_front()
        indicator = indicators.IGD(pf=reference)
        return indicator.do(pf)

    def pareto_plot(self):
        results = self.results()
        y, best_x, best_y = results["y"], results["best_x"], results["best_y"]

        plt.plot(y[:, 0], y[:, 1], "b.", label="evaluated points")
        plt.plot(best_x[:, 0], best_y[:, 1], "r.", label="best points")

        def zdt1_pareto(n_points=100):
            f = np.zeros([n_points, 2])
            f[:, 0] = np.linspace(0, 1, n_points)
            f[:, 1] = 1.0 - np.sqrt(f[:, 0])
            return f

        y_true = zdt1_pareto()
        plt.plot(y_true[:, 0], y_true[:, 1], "k-", label="True Pareto")
        plt.legend()

    def dispatch_code_debug(self, inline=True, project_directory=None, python=None):
        from machinable import Project
        from machinable.utils import chmodx
        from machinable.element import _CONNECTIONS as connected_elements

        if project_directory is None:
            project_directory = Project.get().path()
        if python is None:
            python = sys.executable
        lines = ["from machinable import Project, Element, Component"]
        # context
        lines.append(f"Project('{project_directory}').__enter__()")
        for kind, elements in connected_elements.items():
            if kind in ["Project", "Execution"]:
                continue
            for element in elements:
                jn = element.as_json().replace('"', '\\"').replace("'", "\\'")
                lines.append(f"Element.from_json('{jn}').__enter__()")
        # dispatch
        # lines.append("from mpi4py import MPI")
        # lines.append("import debugpy")
        # lines.append("rank = MPI.COMM_WORLD.Get_rank()")
        # lines.append("if rank == 1:")
        # lines.append("    debugpy.listen(5678)")
        # lines.append("    debugpy.wait_for_client()")
        # lines.append("    debugpy.breakpoint()")
        lines.append(
            f"component__ = Component.from_directory('{self.local_directory()}')"
        )
        lines.append("component__.dispatch()")

        # write python script
        dispatch_script = chmodx(
            self.execution.save_file(
                [self.id, "dispatch.py"], "\n".join([f"#!{python}"] + lines)
            )
        )

        return f"\n\ncd {project_directory}\n\nexport PYTHONPATH={project_directory}:$PYTHONPATH\n\nibrun {dispatch_script}"

        if inline:
            code = ";".join(lines)
            return f'{python} -c "{code}"'

        return "\n".join(lines)
