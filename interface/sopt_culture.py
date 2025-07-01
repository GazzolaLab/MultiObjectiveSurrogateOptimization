from typing import Dict, Optional
from interface.sopt import Sopt
from pydantic import Field
from miv_simulator.env import Env
from miv_simulator.optimization import optimization_params
from miv_simulator.utils import from_yaml
import datetime
from mpi4py import MPI
from dmosopt import dmosopt
from dmosopt import config
from machinable.config import to_dict


class Culture(Sopt):
    class Config(Sopt.Config):
        dopt_params: Dict = Field(
            default_factory=lambda: {
                "opt_id": "miv_simulator.optimize_network",
                "obj_fun_init_name": "miv_simulator.optimize_network.init_network_objfun",
                "controller_init_fun_name": "miv_simulator.optimize_network.init_controller",
                "controller_init_fun_args": {
                    "subworld_size": "???",
                    "use_coreneuron": False,
                },
                "reduce_fun_name": "miv_simulator.optimize_network.compute_objectives",
                "problem_parameters": {},
                "feature_dtypes": "benchmarks.culture.feature_dtypes",
            }
        )
        spawn_startup_wait: Optional[int] = 3

    def version_from_protocol(
        self, filepath: str, nprocs_per_worker: int, **network_config
    ):
        run_ts = datetime.datetime.today().strftime("%Y%m%d_%H%M")

        operational_config = from_yaml(filepath)
        operational_config["run_ts"] = run_ts
        operational_config["nprocs_per_worker"] = nprocs_per_worker

        network_config.update(operational_config.get("kwargs", {}))

        env = Env(**network_config)

        objective_names = operational_config["objective_names"]
        param_config_name = operational_config["param_config_name"]
        target_populations = operational_config["target_populations"]

        opt_param_config = optimization_params(
            env.netclamp_config.optimize_parameters,
            target_populations,
            param_config_name=param_config_name,
            phenotype_dict=env.phenotype_ids,
        )

        opt_targets = opt_param_config.opt_targets
        param_names = opt_param_config.param_names
        param_tuples = opt_param_config.param_tuples
        hyperprm_space = {
            param_pattern: [param_tuple.param_range[0], param_tuple.param_range[1]]
            for param_pattern, param_tuple in zip(param_names, param_tuples)
        }

        return {
            "dopt_params": {
                "space": hyperprm_space,
                "objective_names": objective_names,
                "constraint_names": [
                    f"{target_pop_name} positive rate"
                    for target_pop_name in target_populations
                ],
                "reduce_fun_args": (operational_config, opt_targets),
                "controller_init_fun_args": {
                    "subworld_size": nprocs_per_worker,
                },
                "obj_fun_init_args": {
                    "operational_config": operational_config,
                    "opt_targets": opt_targets,
                    "param_tuples": [
                        param_tuple._asdict() for param_tuple in param_tuples
                    ],
                    "param_names": param_names,
                    **network_config,
                },
            },
            "nprocs_per_worker": nprocs_per_worker,
            "worker_debug": True,
        }

    def version_from_config(
        self,
        filepath: str,
        target_populations: Optional[list[str]] = None,
        objective_names: Optional[list[str]] = None,
        nprocs_per_worker: int = 9,
        key: str = "Weight all",
    ):
        run_ts = datetime.datetime.today().strftime("%Y%m%d_%H%M")

        operational_config = {}
        operational_config["run_ts"] = run_ts
        operational_config["nprocs_per_worker"] = nprocs_per_worker

        network_config = {
            "config": filepath,
            "tstop": 1250,
            "checkpoint_interval": 0,
            "v_init": -77.0,
            "dt": 0.025,
            "cleanup": False,
            "verbose": False,
            "cache_queries": True,
            "output_results": False,
            "use_coreneuron": False,
        }
        env = Env(**network_config)

        opt_params = env.netclamp_config.optimize_parameters

        if target_populations is None:
            target_populations = list(opt_params["synaptic"].keys())

        operational_config["target_populations"] = target_populations

        if objective_names is None:
            objective_names = [
                f"{target_pop_name} {field}"
                for target_pop_name in target_populations
                for field in ["firing rate", "fraction active"]
            ]

        operational_config["objective_names"] = objective_names

        opt_param_config = optimization_params(
            opt_params,
            target_populations,
            key,
        )

        opt_targets = opt_param_config.opt_targets
        param_names = opt_param_config.param_names
        param_tuples = opt_param_config.param_tuples

        hyperprm_space = {
            param_pattern: [param_tuple.param_range[0], param_tuple.param_range[1]]
            for param_pattern, param_tuple in zip(param_names, param_tuples)
        }

        return {
            "dopt_params": {
                "space": hyperprm_space,
                "objective_names": objective_names,
                "constraint_names": [
                    f"{target_pop_name} positive rate"
                    for target_pop_name in target_populations
                ],
                "reduce_fun_args": (operational_config, opt_targets),
                "controller_init_fun_args": {
                    "subworld_size": nprocs_per_worker,
                },
                "obj_fun_init_args": {
                    "operational_config": operational_config,
                    "opt_targets": opt_targets,
                    "param_tuples": [
                        param_tuple._asdict() for param_tuple in param_tuples
                    ],
                    "param_names": param_names,
                    **network_config,
                },
            },
            "nprocs_per_worker": nprocs_per_worker,
        }

    def __call__(self) -> None:
        ## HACK: override to ensure the correct parsing of tuples in obj_fun_init_args

        if MPI.COMM_WORLD.Get_rank() == 0:
            params = to_dict(self.config.dopt_params)
            if "file_path" not in params:
                params["file_path"] = self.output_filepath
            if "local_random" not in params and "random_seed" not in params:
                params["random_seed"] = self.seed
            for f in [
                "feature_dtypes",
                "objective_names",
                "constraint_names",
                "metadata",
            ]:
                # users may specify these fields in terms of importable objects
                #  to avoid repetition or use custom types
                if f in params and isinstance(params[f], str):
                    fi = config.import_object_by_path(params[f])
                    if callable(fi):
                        fi = fi(self)
                    params[f] = fi

            # OVERRIDE:
            def _o(p):
                if isinstance(p["sec_type"], list):
                    p["sec_type"] = tuple(p["sec_type"])
                return p

            params["obj_fun_init_args"]["param_tuples"] = [
                _o(param_tuple)
                for param_tuple in params["obj_fun_init_args"]["param_tuples"]
            ]
            # OVERRIDE^

            params = MPI.COMM_WORLD.bcast(params, root=0)
        else:
            params = MPI.COMM_WORLD.bcast(None, root=0)

        run = dmosopt.run(
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

    def compute_context(self):
        context = super().compute_context()

        try:
            del context["config"]["dopt_params"]["controller_init_fun_args"]
        except:
            pass

        try:
            del context["config"]["dopt_params"]["obj_fun_init_args"]
        except:
            pass

        try:
            del context["config"]["dopt_params"]["reduce_fun_args"]
        except:
            pass

        return context
