from machinable import Component
from mpi4py import MPI
from pydantic import BaseModel, Field, field_validator, TypeAdapter
from dmosopt import dmosopt
from machinable.config import to_dict
from typing import Dict, Optional, List, Callable, Literal, Set, Any, Union, Tuple
import copy

class Dmosopt(Component):
    class Config(BaseModel):
        sopt_params: Dict = Field("???")
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
        
        @field_validator('sopt_params')
        @classmethod
        def valid_optimization_settings(cls, params: Dict) -> Dict:
            _t = {
                'opt_id': str,
                'obj_fun_name': Optional[str],
                'obj_fun_module': str,
                'obj_fun_init_name': Optional[str],
                'obj_fun_init_args': Dict,
                'controller_init_fun_module': str,
                'controller_init_fun_name': Optional[str],
                'controller_init_fun_args': Dict,
                'reduce_fun_module': str,
                'reduce_fun_name': Optional[str],
                'reduce_fun_args': Dict,
                'broker_fun_name': Optional[str],
                'broker_module_name': Optional[str],
                # DistOptimizer
                'objective_names': List[str],
                'feature_dtypes': List[Tuple[str, Any]],
                'constraint_names': List[str],
                'n_initial': int,
                'initial_maxiter': int,
                'initial_method': Union[
                    Callable, Literal['glp', 'slh', 'lh', 'mc', 'sobol'], Dict[str, Any]
                ],
                'verbose': bool,
                'problem_ids': Optional[Set],
                'problem_parameters': Optional[Dict],
                'space': Optional[Dict[str, Tuple[Union[int, float], Union[int, float]]]],
                'population_size': int,
                'num_generations': int,
                'resample_fraction': float,
                'distance_metric': Any, #
                'n_epochs': int,
                'save_eval': bool,
                'file_path': Optional[str],
                'save': bool,
                'save_surrogate_evals': bool,
                'save_optimizer_params': bool,
                'metadata': Any,
                'surrogate_method': Literal['gpr', 'egp', 'megp', 'mdgp', 'mdspp', 'vgp', 'svgp', 'spv', 'siv', 'crv'],
                'surrogate_options': Dict,
                'optimizer': Literal['nsga2', 'age', 'smpso', 'cmaes'],
                'optimizer_options': Dict,
                'sensitivity_method': Literal['dgsm', 'fast'],
                'sensitivity_options': Dict,
                'local_random': Any,
                'random_seed': Optional[int],
                'feasibility_model': bool,
                'termination_conditions': Optional[Dict],
                # 
                'di_crossover': Any, #
                'di_mutation': Any, #
            }
            
            payload = copy.deepcopy(to_dict(params))
            
            for k, v in payload.items():
                if k not in _t:
                    raise ValueError(f"Invalid option: {k}")
                TypeAdapter(_t[k]).validate_python(v)
                
            # additional rules
            if (payload.get('random_seed', None) is not None) and (payload.get('local_random', None) is not None):
                raise ValueError(
                    "Both random_seed and local_random are specified! Only one or the other must be specified."
                )
                
            return payload
        

    def __call__(self) -> None:
        params = to_dict(self.config.sopt_params)
        if "file_path" not in params:
            params["file_path"] = self.output_filepath
        dmosopt.run(
            sopt_params=params,
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

    @property
    def output_filepath(self) -> str:
        return self.local_directory("dmosopt.h5")

    def on_write_meta_data(self):
        return MPI.COMM_WORLD.Get_rank() == 0
