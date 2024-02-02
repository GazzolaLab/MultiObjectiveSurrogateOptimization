from typing import Dict
from interface.dmosopt import Dmosopt
from pydantic import Field
from machinable import get

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
                #
                "controller_init_fun_name": "benchmarks.mc.init_controller",
                "controller_init_fun_args": {
                    "subworld_size": 1,
                    "use_coreneuron": USE_CORENEURON,
                },
                # "reduce_fun_name": "miv_simulator.optimization.compute_objectives",
                # "reduce_fun_args": (),
                "problem_parameters": {},
                "space": {
                    "PYR.PYR.apical.AMPA.weight.a": [1, 3],
                    # ...
                },
                "objective_names": [
                    "GC snr",
                    "GC firing rate",
                    "GC fraction active"
                    # ...
                ],
                "feature_dtypes": "benchmarks.mc.feature_dtypes",  # objective_names -> np.float32
                "constraint_names": [
                    "PYR positive rate",
                    # for each target population
                ],
                "optimizer_name": "nsga2",
                # "optimizer_kwargs": {"sampling_method": "sobol"},
                "initial_method": "slh",
                "n_initial": 30,  # times number of parameters
                "initial_maxiter": 0,  # 50
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
