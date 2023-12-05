# %% [markdown]
# # Motoneurons
#
#

# %%
from machinable import get

import numpy as np

N_exp = len([20, 30, 40, 50, 60, 70, 80])

feature_dtypes = [
    (
        "ic_constant_hold",
        np.float32,
    ),
    (
        "ic_constant_rest",
        np.float32,
    ),
    (
        "initial_v_error_hold",
        np.float32,
    ),
    (
        "rn",
        np.float32,
    ),
    (
        "tau",
        np.float32,
    ),
    ("fI", np.dtype([("frequency", float)]), N_exp),
    ("mean_fI_diff", np.float32),
    (
        "ISI",
        np.dtype(
            [
                ("first", float),
                ("last", float),
                ("ratio", float),
                ("mean", float),
                ("std", float),
                ("N", int),
            ]
        ),
        N_exp,
    ),
    ("threshold", np.float32, N_exp),
    ("spike_amplitude", np.float32, N_exp),
]

defaults = {
    "optimizer": {
        "opt_id": "default",
        "obj_fun_init_name": "make_obj_fun",
        "obj_fun_init_args": {
            "template_name": "MN_nrn",
            "feature_dtypes": feature_dtypes,
        },
        "problem_parameters": {
            "soma_f_Caconc": 0.004,
            "soma_alpha_Caconc": 1,
            "soma_kCa_Caconc": 8,
            "dend_f_Caconc": 0.004,
            "dend_alpha_Caconc": 1,
            "dend_kCa_Caconc": 8,
            "global_diam": 5,
            "global_cm": 2,
            "e_pas": -62,
            "pp": 0.1,
            "Ltotal": 120,
        },
        "space": {
            "gc": [0.1, 2],
            "soma_gmax_Na": [0.1, 0.3],
            "soma_gmax_K": [0.01, 0.3],
            "soma_gmax_KCa": [0.0001, 0.01],
            "soma_gmax_CaN": [0.00001, 0.03],
            "soma_g_pas": [0.00001, 0.01],
            "dend_gmax_CaL": [0.00001, 0.001],
            "dend_gmax_CaN": [0.00001, 0.001],
            "dend_gmax_KCa": [0.0001, 0.005],
            "dend_g_pas": [0.00001, 0.01],
            "cm_ratio": [1, 40],
        },
        "objective_names": [
            "rn_error",
            "tau_error",
            "fI_error",
            "spike_amplitude_error",
            "ISI_adaptation_error",
        ],
        "constraint_names": [
            "monotonic_fI",
            "rn_constr",
            "tau_constr",
            "spike_amplitude_constr",
            "first_ISI_constr",
            "ISI_adaptation_constr",
            "pre_spk_count",
            "initial_v_constr",
        ],
        "feature_dtypes": feature_dtypes,
        "optimizer": "nsga2",
        "optimizer_options": {"sampling_method": "sobol"},
        "n_initial": 800,
        "n_epochs": 10,
        "population_size": 400,
        "num_generations": 400,
        "termination_conditions": True,
        "resample_fraction": 1.0,
        "initial_maxiter": 10,
        "initial_method": "slh",
        "surrogate_method": "gpr",
        "surrogate_options": {
            "lengthscale_bounds": (1e-5, 100.0),
            "batch_size": 100,
            "n_iter": 30000,
            "min_elbo_pct_change": 1.0,
            "cuda": False,
        },
        "feasibility_model": False,
        "save": True,
        "save_surrogate_eval": False,
    }
}


# %%

with get(
    "interface.execution.slurm", {"nodes": 32, "ranks": 56, "partition": "normal"}
):
    motoneuron = get(
        "interface.dmosopt",
        [defaults, {}],
    ).launch()
