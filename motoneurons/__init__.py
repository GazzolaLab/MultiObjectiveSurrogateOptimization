import os
import time
from dataclasses import dataclass
from typing import Dict, Union

import numpy as np
import omegaconf
from dmosopt import dmosopt
from machinable import Component
from machinable.config import Field
from miv_simulator.mechanisms import compile_and_load
from mpi4py import MPI
from numpy.random import default_rng


class Motoneurons(Component):
    @dataclass
    class Config:
        protocol: Dict = Field("proto()")
        target_namespace: str = None
        model_variant: str = "default"
        num_epochs: int = 10
        num_initial: int = 800
        population_size: int = 400
        num_generations: int = 400
        seed: int = None
        initial_method: str = "slh"
        optimizer: str = "nsga2"
        resample_fraction: float = 1.0
        surrogate_method: str = "gpr"
        surrogate_n_iter: int = 30000
        save_surrogate_eval: bool = False
        gpytorch_cuda: bool = False
        sensitivity: bool = False
        feasibility: bool = False

    @property
    def output_filepath(self):
        return self.local_directory("dmosopt.h5")

    def on_instantiate(self):
        compile_and_load(os.path.join(os.path.dirname(__file__), "mechanisms"))

    def on_write_meta_data(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        return rank == 0

    def __call__(self):
        import motoneurons.ephys_utils as ephys
        from motoneurons.neuron_utils import load_template
        from motoneurons.protocol import ExperimentalProtocol

        comm = MPI.COMM_WORLD

        local_random = None
        if self.config.seed is not None:
            local_random = default_rng(seed=self.config.seed)

        celltype = self.config.protocol["Celltype"]
        template_dict = self.config.protocol.get("Template", None)
        template_name = None
        template_file = None
        template = None
        if template_dict is None:
            template_name = "MN_nrn"
            template_file = "MN_nrn.hoc"
        else:
            if self.config.model_variant in template_dict:
                template_name = template_dict[self.config.model_variant]["name"]
                template_file = template_dict[self.config.model_variant].get(
                    "file", None
                )
            else:
                raise RuntimeError(f"Unknown model variant {self.config.model_variant}")
        template = load_template(
            template_name, os.path.join(os.path.dirname(__file__), template_file)
        )

        N_exp = len(self.config.protocol["Targets"]["f_I"]["I"])
        if self.config.target_namespace is not None:
            N_exp = len(
                self.config.protocol["Target namespaces"][self.config.target_namespace][
                    "f_I"
                ]["I"]
            )
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
            ("fI", ephys.fi_value_dtype, N_exp),
            ("mean_fI_diff", np.float32),
            ("ISI", ephys.isi_value_dtype, N_exp),
            ("threshold", np.float32, N_exp),
            ("spike_amplitude", np.float32, N_exp),
        ]

        problem_parameters = self.config.protocol["Parameters"]
        variant_parameters_dict = self.config.protocol.get("Variant Parameters", {})
        if self.config.model_variant in variant_parameters_dict:
            variant_parameters = variant_parameters_dict[self.config.model_variant]
            for k in variant_parameters:
                problem_parameters[k] = variant_parameters[k]

        space = self.config.protocol["Space"]
        variant_space_dict = self.config.protocol.get("Variant Space", {})
        if self.config.model_variant in variant_space_dict:
            variant_space = variant_space_dict[self.config.model_variant]
            for k in variant_space:
                space[k] = variant_space[k]

        space_sensitivity = None
        if self.config.sensitivity:
            space_sensitivity = self.config.protocol.get("Space Sensitivity", None)
            variant_sensitivity_dict = self.config.protocol.get(
                "Variant Space Sensitivity", {}
            )

            if self.config.model_variant in variant_sensitivity_dict:
                variant_sensitivity = variant_sensitivity_dict[
                    self.config.model_variant
                ]
                for k in variant_sensitivity:
                    space_sensitivity[k] = variant_sensitivity[k]

        objective_names = [
            "rn_error",
            "tau_error",
            "fI_error",
            "spike_amplitude_error",
            "ISI_adaptation_error",
        ]
        constraint_names = [
            "monotonic_fI",
            "rn_constr",
            "tau_constr",
            "spike_amplitude_constr",
            "first_ISI_constr",
            "ISI_adaptation_constr",
            "pre_spk_count",
            "initial_v_constr",
        ]

        exp_protocol = ExperimentalProtocol(
            self.config.protocol, target_namespace=self.config.target_namespace
        )

        N_spk_amp = min(
            len(exp_protocol.exp_i_lb_spk_amp), len(exp_protocol.exp_i_inj_amp_f_I)
        )
        N_spk_adpt = min(
            len(exp_protocol.exp_i_lb_spk_adaptation),
            len(exp_protocol.exp_i_inj_amp_f_I),
        )
        obj_targets = {
            "rn": (np.asarray(exp_protocol.target_rn, dtype=np.float32), np.float32, 2),
            "tau": (
                np.asarray(exp_protocol.target_tau, dtype=np.float32),
                np.float32,
                2,
            ),
            "ISI_adaptation": (
                np.row_stack(
                    (
                        exp_protocol.exp_i_inj_amp_f_I[:N_spk_adpt],
                        exp_protocol.exp_i_lb_spk_adaptation[:N_spk_adpt],
                        exp_protocol.exp_i_ub_spk_adaptation[:N_spk_adpt],
                    )
                ),
                np.float32,
                (3, N_exp),
            ),
            "fI": (
                np.row_stack(
                    (
                        exp_protocol.exp_i_inj_amp_f_I,
                        exp_protocol.exp_i_lb_rate_f_I,
                        exp_protocol.exp_i_ub_rate_f_I,
                    )
                ),
                np.float32,
                (3, N_exp),
            ),
            "spike_amplitude": (
                np.row_stack(
                    (
                        exp_protocol.exp_i_inj_amp_f_I[:N_spk_amp],
                        exp_protocol.exp_i_lb_spk_amp[:N_spk_amp],
                        exp_protocol.exp_i_ub_spk_amp[:N_spk_amp],
                    )
                ),
                np.float32,
                (3, N_exp),
            ),
        }
        problem_metadata = np.array(
            [tuple((obj_targets[k][0] for k in sorted(obj_targets)))],
            dtype=[
                (f"{k}_target", obj_targets[k][1], obj_targets[k][2])
                for k in sorted(obj_targets)
            ],
        )

        # Create an optimizer
        dmosopt_params = {
            "opt_id": f"dmosopt_{celltype}_neuron",
            "obj_fun_init_name": "make_obj_fun",
            "obj_fun_init_module": "motoneurons.objective",
            "obj_fun_init_args": {
                "protocol_config_dict": self.config.protocol,
                "feature_dtypes": feature_dtypes,
                "template_name": template_name,
                "target_namespace": self.config.target_namespace,
            },
            "problem_parameters": problem_parameters,
            "space": space,
            "objective_names": objective_names,
            "constraint_names": constraint_names,
            "feature_dtypes": feature_dtypes,
            "optimizer": self.config.optimizer,
            "optimizer_options": {"sampling_method": "sobol"},
            "n_initial": self.config.num_initial,
            "n_epochs": self.config.num_epochs,
            "population_size": self.config.population_size,
            "num_generations": self.config.num_generations,
            "termination_conditions": True,
            "resample_fraction": self.config.resample_fraction,
            "initial_maxiter": 10,
            "initial_method": self.config.initial_method,
            "surrogate_method": self.config.surrogate_method,
            "surrogate_options": {
                "lengthscale_bounds": (1e-5, 100.0),
                "batch_size": 100,
                "n_iter": self.config.surrogate_n_iter,
                "min_elbo_pct_change": 1.0,
                "cuda": self.config.gpytorch_cuda,
            },
            "feasibility_model": self.config.feasibility,
            "file_path": self.output_filepath,
            "save": True,
            "save_surrogate_eval": self.config.save_surrogate_eval,
            "metadata": problem_metadata,
            "local_random": local_random,
        }

        if space_sensitivity is not None:
            dmosopt_params["di_crossover"] = space_sensitivity
            dmosopt_params["di_mutation"] = space_sensitivity

        def _untyped(dict_like):
            if isinstance(dict_like, (omegaconf.DictConfig, omegaconf.ListConfig)):
                return omegaconf.OmegaConf.to_container(dict_like)
            if isinstance(dict_like, (list, tuple)):
                return dict_like.__class__([k for k in dict_like])
            if not isinstance(dict_like, dict):
                return dict_like
            return {k: _untyped(v) for k, v in dict_like.items()}

        best = dmosopt.run(_untyped(dmosopt_params), verbose=True)

        self.save_file("best.p", best)

    def plot(self):
        import matplotlib.pyplot as plt

        from motoneurons.neuron_utils import load_template, run_iclamp

        data = self.load_file("best.p")

        if data is None:
            raise ValueError("Data not available")

        bestx, besty = data

        bestx_dict = dict(bestx)

        space = self.config.protocol["Space"]
        variant_space_dict = self.config.protocol.get("Variant Space", {})
        if self.config.model_variant in variant_space_dict:
            variant_space = variant_space_dict[self.config.model_variant]
            for k in variant_space:
                space[k] = variant_space[k]

        problem_parameters = self.config.protocol["Parameters"]
        variant_parameters_dict = self.config.protocol.get("Variant Parameters", {})
        if self.config.model_variant in variant_parameters_dict:
            variant_parameters = variant_parameters_dict[self.config.model_variant]
            for k in variant_parameters:
                problem_parameters[k] = variant_parameters[k]

        param_dict = {k: bestx_dict[k][0] for k in space}
        for p in problem_parameters:
            param_dict[p] = problem_parameters[p]

        template_dict = self.config.protocol.get("Template", None)
        template_name = None
        template_file = None
        template = None
        if template_dict is None:
            template_name = "MN_nrn"
            template_file = "MN_nrn.hoc"
        else:
            if self.config.model_variant in template_dict:
                template_name = template_dict[self.config.model_variant]["name"]
                template_file = template_dict[self.config.model_variant].get(
                    "file", None
                )
            else:
                raise RuntimeError(f"Unknown model variant {self.config.model_variant}")
        template = load_template(
            template_name, os.path.join(os.path.dirname(__file__), template_file)
        )

        cell = template(param_dict)

        vec_t, vec_v = run_iclamp(cell, 0.15, 500.0, 1000.0)

        plt.plot(vec_t, vec_v, linewidth=3, color="r")
        plt.xlabel("Time (ms)")
        plt.ylabel("V (mV)")

        fp = self.local_directory("plot.png")
        plt.savefig(fp)

        print(fp)

        return fp

    def config_proto(self):
        return {
            "Celltype": "Motoneuron",
            "Numerics": {
                "adaptive": False,
                "use_coreneuron": True,
                "t0": 0,
                "tstop": 2000,
                "dt": 0.0125,
                "record_dt": 0.01,
                "v_init": -60,
            },
            "Record": {"soma": ["V"], "dend": ["V"]},
            "Targets": {
                "Rin": {
                    "I": [-100],
                    "I_factor": 0.001,
                    "upper": [598],
                    "lower": [540],
                    "t": [250, 1250],
                },
                "tau0": {
                    "I": [-0.1],
                    "upper": [19.375],
                    "lower": [16.308],
                    "t": [250, 1250],
                },
                "threshold": -37,
                "V_hold": {"val": -60, "I": 0},
                "V_rest": {"val": -57.4, "I": 0},
                "f_I": {
                    "I": [20, 30, 40, 50, 60, 70, 80],
                    "I_factor": 0.001,
                    "mean": [3.88, 9.09, 11.75, 14.29, 15.96, 16.58, 18.25],
                    "t": [500, 1500],
                },
                "spike_amp": {
                    "upper": [80, 80, 80, 80, 80, 80, 80],
                    "lower": [60, 60, 60, 60, 60, 60, 60],
                },
                "spike_adaptation": {
                    "upper": [1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6],
                    "lower": [1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2],
                    "t": [500, 1500],
                },
            },
            "Parameters": {
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
            "Space": {
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
        }
