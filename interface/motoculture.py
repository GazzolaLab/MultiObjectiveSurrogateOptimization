from typing import Dict
from interface.dmosopt import Dmosopt
from pydantic import Field
from machinable import get
import yaml
from miv_simulator.optimization import optimization_params

# isort: off

with get("interface.require"):
    from benchmarks.motoneuron_culture.simulation import microcircuit as network

USE_CORENEURON = False


class Motoculture(Dmosopt):
    class Config(Dmosopt.Config):
        dopt_params: Dict = Field(
            default_factory=lambda: {
                "opt_id": "default",
                "obj_fun_init_name": "benchmarks.mc.init_network_objfun",
                "obj_fun_init_args": {
                    "cells": network.neural_h5.graph.cells_filepath,
                    "connections": network.neural_h5.graph.connections_filepath,
                    "cell_types": network.config.cell_types,
                    "synapses": network.config.synapses,
                    "mechanisms": network.MECHANISMS,
                    "templates": f"{network.SOURCE}/templates",
                    "use_coreneuron": USE_CORENEURON,
                },
                "controller_init_fun_name": "miv_simulator.optimize_network.init_controller",
                "controller_init_fun_args": {
                    "subworld_size": 1,
                    "use_coreneuron": USE_CORENEURON,
                },
                "reduce_fun_name": "miv_simulator.optimize_network.compute_objectives",
                "reduce_fun_args": (),  #  TODO
                "problem_parameters": {},
                # "space": {},
                "objective_names": [
                    "GC snr",
                    "GC firing rate",
                    "GC fraction active",
                    # ... TODO
                ],
                "feature_dtypes": "benchmarks.mc.feature_dtypes",
                # "constraint_names": [],
                "optimizer_name": "nsga2",
                # "optimizer_kwargs": {},
                "initial_method": "slh",
                "n_initial": 30,  # times number of parameters
                "initial_maxiter": 50,
                "n_epochs": 10,
                "population_size": 100,
                "num_generations": 200,
                "resample_fraction": 1.0,  # times the population_size
                "surrogate_method_name": "gpr",  # megp , gpr, None,
                "surrogate_method_kwargs": {},
                # "mutation_rate": mutation_rate,
                # "termination_conditions": True,
                "feasibility_method_name": None,
                "feasibility_method_kwargs": {},
                "save": True,
                "save_surrogate_evals": True,
                # "save_eval": 5,
            }
        )

    def version_from_protocol(
        self,
        filepath: str = "benchmarks/motoneuron_culture/config/Network_Clamp.yaml",
        name: str = "Weight all",
    ):
        with open(filepath) as f:
            protocol_config_dict = yaml.load(f, Loader=yaml.FullLoader)
            protocol_config_dict["synaptic"] = protocol_config_dict.pop(
                "Synaptic Optimization"
            )

        target_populations = list(protocol_config_dict["synaptic"].keys())

        opt_param_config = optimization_params(
            protocol_config_dict,
            target_populations,
            name,
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
                "constraint_names": [
                    f"{target_pop_name} positive rate"
                    for target_pop_name in target_populations
                ],
            }
        }
