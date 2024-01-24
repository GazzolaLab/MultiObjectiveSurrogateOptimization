# %% [markdown]
# # Motoneuron culture
#
#

# %%
import os
from machinable import get
# isort: off

get("machinable.index", os.environ["STORAGE"]).__enter__()

with get('interface.require'):
    from benchmarks.motoneuron_culture.simulation import microcircuit as network

print("Network:", network.neural_h5.files())

# %%

USE_CORENEURON = False
motoculture = [
    "interface.dmosopt",
    {
        "dopt_params": {
            "opt_id": "default",
            "obj_fun_init_name": "benchmarks.mc.init_network_objfun",
            "obj_fun_init_args": {
                "cells": network.neural_h5.graph.cells_filepath,
                "connections": network.neural_h5.graph.connections,
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
            "n_epochs": 1,
            "population_size": 100,
            "num_generations": 200,
            "resample_fraction": 1.0,  # times the population_size
            "surrogate_method_name": "gpr",  # megp , gpr, None,
            "surrogate_method_kwargs": {},
            # "mutation_rate": mutation_rate,
            # "termination_conditions": True,
            # "sensitivity_method": "dgsm",
            "feasibility_model": False,
            "save": True,
            "save_surrogate_evals": True,
            # "save_eval": 5,
        }
    },
]

# %%

with get("interface.execution.local", {"mpi": "ibrun"}):
    motoculture_ = get(motoculture).launch()
