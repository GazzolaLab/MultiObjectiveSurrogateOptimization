import os

from interface.sopt import Sopt
from typing import Dict, Optional
from pydantic import Field
from miv_simulator.mechanisms import compile
import yaml
import numpy as np


class SoptWc(Sopt):
    class Config(Sopt.Config):
        dopt_params: Dict = Field(
            default_factory=lambda: {
                "opt_id": "dmosopt_wilson_cowan",
                "optimizer_name": "nsga2",
                "initial_method": "mc",
                "n_initial": 10,
                "initial_maxiter": 10,
                "n_epochs": 10,
                "population_size": 100,
                "num_generations": 10,
                "termination_conditions": True,
                "resample_fraction": 1.0,
                "surrogate_method_name": None,
                "surrogate_method_kwargs": {},
                "feasibility_method_name": None,
                "feasibility_method_kwargs": {},
                "save": True,
                "save_surrogate_evals": True,
                "obj_fun_init_name": "benchmarks.cortical_culture.e_i_system.obj_fun_init",
                "obj_fun_init_args": {},
                "objective_names": [
                    "Delta_mean",
                    "Theta_mean",
                    "Alpha_mean",
                    "Beta_mean",
                    "Gamma_mean",
                    "Delta_std",
                    "Theta_std",
                    "Alpha_std",
                    "Beta_std",
                    "Gamma_std",
                ],
                "feature_dtypes": "benchmarks.cortical_culture.e_i_system.feature_dtypes",
                "space": {
                    "E_E_weight": [0.001, 25.0],
                    "E_I_weight": [0.001, 25.0],
                    "I_E_weight": [0.001, 25.0],
                    "I_I_weight": [0.001, 25.0],
                    "diffusion_strength": [0.001, 50.0],
                    "E_theta": [0.001, 10.0],
                    "I_theta": [0.001, 10.0],
                    "E_tau": [0.001, 10.0],
                    "I_tau": [0.001, 10.0],
                    "E_gain": [0.001, 50.0],
                    "I_gain": [0.001, 50.0],
                    "E_E_scale": [0.001, 50.0],
                    "I_E_scale": [0.001, 50.0],
                    "E_I_scale": [0.001, 50.0],
                    "I_I_scale": [0.001, 50.0],
                },
                "problem_parameters": {},
            }
        )
