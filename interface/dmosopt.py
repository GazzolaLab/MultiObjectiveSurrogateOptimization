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
from dmosopt import indicators
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
                "objective_names": Union[str, List[str]],
                "feature_dtypes": str,
                "constraint_names": Union[str, List[str]],
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
                "optimizer_name": Union[Literal["nsga2", "age", "smpso", "cmaes"], str],
                "optimizer_kwargs": Union[Dict, List[Dict]],
                "sensitivity_method_name": Literal["dgsm", "fast"],
                "sensitivity_method_kwargs": Dict,
                "local_random": Any,
                "random_seed": Optional[int],
                "feasibility_method_name": Optional[str],
                "feasibility_method_kwargs": Dict,
                "termination_conditions": Union[bool, Dict, None],
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
                (
                    "feasibility_method_name",
                    config.default_feasibility_methods,
                    "feasibility_method_kwargs",
                ),
                ("surrogate_custom_training", {}, None),
                ("optimizer_name", config.default_optimizers, None),
                (
                    "sensitivity_method_name",
                    config.default_sa_methods,
                    "sensitivity_method_kwargs",
                ),
                ("feature_dtypes", {}, None),
                ("objective_names", {}, None),
                ("constraint_names", {}, None),
                ("metadata", {}, None),
            ]:
                if isinstance(target := payload.get(path, None), str):
                    if target in alias:
                        target = alias[target]
                    try:
                        obj = config.import_object_by_path(target)
                    except ImportError as _ex:
                        raise ValueError(
                            f"Could not resolve import path '{target}' for '{path}': {_ex}"
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
        for f in ["feature_dtypes", "objective_names", "constraint_names", "metadata"]:
            # users may specify these fields in terms of importable objects
            #  to avoid repetition or use custom types
            if f in params and isinstance(params[f], str):
                fi = config.import_object_by_path(params[f])
                if callable(fi):
                    fi = fi(self)
                params[f] = fi
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

    def parameter_vector_to_dict(self, x, include_constants=True):
        constants = {}
        if include_constants:
            constants = self.config.dopt_params.problem_parameters
        return {
            **constants,
            **{k: x[n] for n, k in enumerate(self.config.dopt_params.space.keys())},
        }

    def evaluate_objective_at(self, x):
        import logging

        logging.basicConfig(level=logging.INFO)
        if "obj_fun_init_name" in self.config.dopt_params:
            obj_fun = config.import_object_by_path(
                self.config.dopt_params.obj_fun_init_name
            )(**self.config.dopt_params.obj_fun_init_args)
        else:
            obj_fun = config.import_object_by_path(self.config.dopt_params.obj_fun_name)

        return obj_fun(self.parameter_vector_to_dict(x))

    @property
    def output_filepath(self) -> str:
        return os.path.abspath(self.local_directory("dmosopt.h5"))

    def on_write_meta_data(self):
        return MPI.COMM_WORLD.Get_rank() == 0

    def load_h5(
        self,
        filepath: Optional[str] = None,
        opt_id: Optional[str] = None,
        problem_id: int = 0,
    ):
        if filepath is None:
            filepath = self.output_filepath

        if opt_id is None:
            opt_id = self.config.dopt_params.opt_id

        with h5py.File(filepath, "r") as h5:
            # constaints
            if "constraint_names" in self.config.dopt_params:
                constraint_enum = h5py.check_enum_dtype(
                    h5[f"{opt_id}/constraint_enum"].dtype
                )
                constraint_enum_T = {v: k for k, v in constraint_enum.items()}
                constraint_names = [
                    constraint_enum_T[s[0]]
                    for s in iter(h5[f"{opt_id}/constraint_spec"])
                ]
                constraints = pd.DataFrame(
                    h5[f"{opt_id}/{problem_id}/constraints"][:],
                    columns=constraint_names,
                )
            else:
                constraints = None

            # epochs
            epochs = h5[f"{opt_id}/{problem_id}/epochs"][:]

            # features
            feature_enum = h5py.check_enum_dtype(h5[f"{opt_id}/feature_enum"].dtype)
            feature_enum_T = {v: k for k, v in feature_enum.items()}
            feature_names = [
                feature_enum_T[s[0]] for s in iter(h5[f"{opt_id}/feature_spec"])
            ]
            features = h5[f"{opt_id}/{problem_id}/features"][:]

            # objectives
            objective_enum = h5py.check_enum_dtype(h5[f"{opt_id}/objective_enum"].dtype)
            objective_enum_T = {v: k for k, v in objective_enum.items()}
            objective_names = [
                objective_enum_T[s[0]] for s in iter(h5[f"{opt_id}/objective_spec"])
            ]
            objectives = pd.DataFrame(
                h5[f"{opt_id}/{problem_id}/objectives"][:], columns=objective_names
            )

            # parameters
            parameter_enum = h5py.check_enum_dtype(h5[f"{opt_id}/parameter_enum"].dtype)
            parameter_enum_T = {v: k for k, v in parameter_enum.items()}
            parameter_names = [
                parameter_enum_T[s[0]] for s in iter(h5[f"{opt_id}/parameter_spec"])
            ]
            parameters = pd.DataFrame(
                h5[f"{opt_id}/{problem_id}/parameters"][:], columns=parameter_names
            )

            # predictions
            predictions = pd.DataFrame(
                h5[f"{opt_id}/{problem_id}/predictions"][:], columns=objective_names
            )

            # metadata
            metadata = None
            if f"/{opt_id}/metadata" in h5:
                metadata = h5[f"/{opt_id}/metadata"][:]

        return {
            "constraints": constraints,
            "epochs": epochs,
            "features": features,
            "objectives": objectives,
            "parameters": parameters,
            "predictions": predictions,
            "metadata": metadata,
        }

    def get_best(self, region=None, sort_by="-np.std(y, axis=1)"):
        if region is None:
            region = slice(None)
        data = self.load_h5()
        X = data["parameters"].to_numpy()[region]
        if data["constraints"] is not None:
            C = data["constraints"].to_numpy()[region]
        else:
            C = None
        objectives = data["objectives"].to_numpy()[region]
        f = data["features"][region]
        best_x, best_y, best_f, best_c, best_epoch, perm = get_best(
            X, objectives, f, C, None, None
        )

        if isinstance(sort_by, str):
            context = {
                "reduced": None,
                "x": best_x,
                "y": best_y,
                "f": best_f,
                "c": best_c,
                "np": np,
            }
            exec(f"reduced={sort_by}", context)
            sort_by = np.argsort(context["reduced"])

        best = {"x": best_x, "y": best_y, "f": best_f, "c": best_c, "epoch": best_epoch}

        # apply sort
        if sort_by is not None:
            for k in best.keys():
                if best[k] is not None:
                    best[k] = best[k][sort_by]

        return best

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
