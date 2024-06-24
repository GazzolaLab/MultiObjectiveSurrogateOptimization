import os

from interface.sopt import Sopt
from typing import Dict, Optional
from pydantic import Field
from miv_simulator.mechanisms import compile
import yaml
import numpy as np


class Modeling(Sopt):
    class Config(Sopt.Config):
        dopt_params: Dict = Field(
            default_factory=lambda: {
                "opt_id": "default",
                "feature_dtypes": "benchmarks.modeling.objective.feature_dtypes",
                "optimizer_name": "nsga2",
                "initial_method": "slh",
                "n_initial": 100,
                "initial_maxiter": 10,
                "n_epochs": 10,
                "population_size": 400,
                "num_generations": 400,
                "termination_conditions": True,
                "resample_fraction": 1.0,
                "surrogate_method_name": None,
                "surrogate_method_kwargs": {},
                "feasibility_method_name": None,
                "feasibility_method_kwargs": {},
                "save": True,
                "save_surrogate_evals": True,
            }
        )
        nodes: str = "20"

    def version_from_protocol(
        self,
        filepath: str = "benchmarks/motoneuron_modeling/config/motoneuron.yaml",
        model_variant: str = "default",
        target_namespace: Optional[str] = None,
        template_path: Optional[str] = None,
        mechanisms_path: Optional[str] = None,
    ):
        source = os.path.dirname(os.path.dirname(filepath))
        if template_path is None:
            template_path = source
        if mechanisms_path is None:
            mechanisms_path = os.path.join(source, "mechanisms")

        with open(filepath) as f:
            protocol_config_dict = yaml.load(f, Loader=yaml.FullLoader)

        if "best" in protocol_config_dict:
            del protocol_config_dict["best"]

        if "p0" in protocol_config_dict:
            del protocol_config_dict["p0"]

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
                "obj_fun_init_name": "benchmarks.modeling.objective.obj_fun_init_from_protocol",
                "obj_fun_init_args": {
                    "protocol_config_dict": protocol_config_dict,
                    "template_name": template_name,
                    "template_path": template_path,
                    "mechanisms": compile(mechanisms_path, recursive=False),
                    "target_namespace": target_namespace,
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
                "problem_parameters": problem_parameters,
                "space": space,
                "feature_dtypes": "benchmarks.modeling.objective.feature_dtypes_from_protocol",
                "metadata": "benchmarks.modeling.objective.metadata_from_protocol",
            }
        }

    def plot_features(
        self, feature_selection="-np.std(y, axis=1)", metadata=None, fontsize="large"
    ):
        from matplotlib import pyplot as plt

        if feature_selection is None or isinstance(feature_selection, str):
            feature_selection = self.get_best(sort_by=feature_selection)["f"][-1]
        elif isinstance(feature_selection, int):
            feature_selection = self.get_best()["f"][feature_selection]

        if metadata is None:
            metadata = self.load_h5()["metadata"]

        # gridspec inside gridspec
        fig = plt.figure(constrained_layout=True, figsize=(15, 4))
        subfigs = fig.subfigures(1, 2, wspace=0.07, width_ratios=[1.1, 2])

        axsLeft = subfigs[0].subplots(3, 1, sharey=False)
        subfigs[0].set_facecolor("0.9")

        ax = axsLeft[0]

        rn_range = metadata["rn_target"].reshape((-1,))
        ax.barh(0.5, rn_range[1] - rn_range[0], height=0.3, left=rn_range[0])
        ax.plot(
            feature_selection["rn"],
            0.5,
            linestyle="",
            markersize=10,
            marker="o",
            color="#ff6600",
            label="Rin",
            markeredgecolor="k",
        )
        # ax.set_xlim(200, 600)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("Input resistance [MOhm]", fontsize=fontsize)
        ax.tick_params(axis="x", labelsize=fontsize)

        ax = axsLeft[1]
        tau_range = metadata["tau_target"].reshape((-1,))
        ax.barh(0.5, tau_range[1] - tau_range[0], height=0.3, left=tau_range[0])
        ax.plot(
            feature_selection["tau"],
            0.5,
            linestyle="",
            markersize=10,
            marker="o",
            color="#ff6600",
            label="Rin",
            markeredgecolor="k",
        )
        # ax.set_xlim(0, 100)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("Membrane time constant [ms]", fontsize=fontsize)
        ax.tick_params(axis="x", labelsize=fontsize)

        ax = axsLeft[2]
        # threshold_range = metadata['threshold_target'].reshape((-1,))
        # ax.barh(0.5, threshold_range[1]-threshold_range[0], height=0.3, left=threshold_range[0])
        ax.vlines(
            self.config.dopt_params.obj_fun_init_args.protocol_config_dict.Targets.threshold,
            0,
            1,
            linestyle="-",
            linewidth=2,
        )
        ax.plot(
            np.mean(feature_selection["threshold"]),
            0.5,
            linestyle="",
            markersize=10,
            marker="o",
            color="#ff6600",
            label="threshold",
            markeredgecolor="k",
        )
        # ax.set_xlim(-80, -20)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("Spike threshold [mV]", fontsize=fontsize)
        ax.tick_params(axis="x", labelsize=fontsize)

        # subfigs[0].suptitle('Left plots', fontsize='x-large')
        axsRight = subfigs[1].subplots(
            1, 2, sharex=False, gridspec_kw={"width_ratios": [1, 1]}
        )
        # subfigs[1].set_facecolor('0.9')

        inj_amp_fI = metadata["fI_target"][0][0]
        target_fI_lb = metadata["fI_target"][0][1]
        target_fI_ub = metadata["fI_target"][0][2]

        ax = axsRight[0]

        ax.plot(
            inj_amp_fI.astype("str"),
            target_fI_lb,
            marker="x",
        )
        ax.plot(
            inj_amp_fI.astype("str"),
            feature_selection["fI"],
            linestyle="",
            markersize=10,
            marker="o",
            color="#ff6600",
            label="threshold",
            markeredgecolor="k",
        )

        ax.set_title("Frequency-current relationship", fontsize=fontsize)
        ax.set_xlabel("Injected current [nA]", fontsize=fontsize)
        ax.set_ylabel("Frequency [Hz]", fontsize=fontsize)
        ax.tick_params(axis="x", labelsize="x-small")
        ax.tick_params(axis="y", labelsize="x-small")

        inj_amp_ISI_adaptation = metadata["fI_target"][0][0]
        target_ISI_adaptation_lb = metadata["ISI_adaptation_target"][0][1]
        target_ISI_adaptation_ub = metadata["ISI_adaptation_target"][0][2]
        ax = axsRight[1]
        if (target_ISI_adaptation_ub == target_ISI_adaptation_lb).all():
            ax.axhline(
                target_ISI_adaptation_ub[0],
                linestyle="-",
                linewidth=2,
            )
        else:
            ax.bar(
                inj_amp_ISI_adaptation.astype("str"),
                height=target_ISI_adaptation_ub - target_ISI_adaptation_lb,
                bottom=target_ISI_adaptation_lb,
                width=0.3,
            )
        ax.plot(
            inj_amp_ISI_adaptation.astype("str"),
            feature_selection["ISI"]["ratio"],
            linestyle="",
            markersize=10,
            marker="o",
            color="#ff6600",
            label="adaptation",
            markeredgecolor="k",
        )
        ax.set_title("ISI adaptation", fontsize=fontsize)
        ax.set_xlabel("Injected current [nA]", fontsize=fontsize)
        ax.set_ylabel("ISI ratio last/first", fontsize=fontsize)
        ax.tick_params(axis="x", labelsize="x-small")
        ax.tick_params(axis="y", labelsize="x-small")

        import matplotlib.lines as mlines

        target = mlines.Line2D([], [], color="#1f77b4", marker="s", ls="", label="Target")
        solution = mlines.Line2D([], [], color="#ff6600", marker="o", ls="", label="Solution")

        plt.legend(handles=[target, solution])

        return fig

    def compute_context(self):
        context = super().compute_context()

        dc = context["config"]["dopt_params"]

        try:
            del dc["obj_fun_init_args"]["mechanisms"]
        except:
            pass

        try:
            del dc["obj_fun_init_args"]["template_path"]
        except:
            pass

        try:
            del dc["obj_fun_init_args"]["protocol_config_dict"]
        except:
            pass

        try:
            del context["config"]["nodes"]
        except:
            pass

        return context
