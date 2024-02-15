import os

from interface.dmosopt import Dmosopt
from typing import Dict, Optional
from pydantic import Field
from miv_simulator.mechanisms import compile
import yaml

MECHANISMS = compile(
    "./benchmarks/motoneuron_modeling/mechanisms",
    os.path.expandvars("$SCRATCH/mechanisms"),
    recursive=False,
    return_hash=True,
)


class Motoneuron(Dmosopt):
    class Config(Dmosopt.Config):
        dopt_params: Dict = Field(
            default_factory=lambda: {
                "opt_id": "default",
                "obj_fun_init_name": "benchmarks.mn.make_obj_fun",
                "obj_fun_init_args": {
                    "template_name": "MN_nrn",
                    "mechanisms": MECHANISMS,
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
                "feature_dtypes": "benchmarks.mn.feature_dtypes",
                "optimizer_name": "nsga2",
                # "optimizer_kwargs": {"sampling_method": "sobol"},
                "initial_method": "slh",
                "n_initial": 800,  # times number of parameters
                "initial_maxiter": 0,
                "n_epochs": 10,
                "population_size": 400,
                "num_generations": 400,
                "termination_conditions": True,
                "resample_fraction": 1.0,  # times the population_size
                "surrogate_method_name": None,  # megp , gpr, None,
                "surrogate_method_kwargs": {},
                "feasibility_method_name": None,
                "feasibility_method_kwargs": {},
                "save": True,
                "save_surrogate_evals": True,
            }
        )


    def version_from_protocol(self, filepath: str = 'benchmarks/motoneuron_modeling/config/motoneuron.yaml', model_variant: str = "default", target_namespace: Optional[str] = None):
        with open(filepath) as f:
            protocol_config_dict = yaml.load(f, Loader=yaml.FullLoader)
        
        if 'best' in protocol_config_dict:
            del protocol_config_dict['best']
            
        celltype = protocol_config_dict["Celltype"]
        template_dict = protocol_config_dict.get("Template", None)
        template_name = None
        if template_dict is None:
            template_name = "MN_nrn"
        else:
            if model_variant in template_dict:
                template_name = template_dict[model_variant]["name"]
            else:
                raise ValueError(f"Unknown model variant {model_variant}")
        
        problem_parameters = protocol_config_dict["Parameters"]
        variant_parameters_dict = protocol_config_dict.get("Variant Parameters", {})
        if model_variant in variant_parameters_dict:
            variant_parameters = variant_parameters_dict[model_variant]
            for k in variant_parameters:
                problem_parameters[k] = variant_parameters[k]
                
        space = protocol_config_dict["Space"]
        variant_space_dict = protocol_config_dict.get("Variant Space", {})
        if model_variant in variant_space_dict:
            variant_space = variant_space_dict[model_variant]
            for k in variant_space:
                space[k] = variant_space[k]

        return {
            "dopt_params": {
                "opt_id": f"dmosopt_{celltype}_neuron",
                "obj_fun_init_name": "benchmarks.mn.protocol_obj_fun_init_adapter",
                "obj_fun_init_args": {
                    "protocol_config_dict": protocol_config_dict,
                    "model_variant": model_variant,
                    "target_namespace": target_namespace,
                    "template_name": template_name,
                },
                "problem_parameters": problem_parameters,
                "space": space,
                "feature_dtypes": "benchmarks.mn.feature_dtypes_from_protocol",
                "initial_maxiter": 10,
                "surrogate_method_name": "gpr",
                "metadata": "benchmarks.mn.metadata_from_protocol",
            }
        }