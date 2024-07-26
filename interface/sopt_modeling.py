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
                "population_size": 300,
                "num_generations": 50,
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
                    "mechanisms": os.path.relpath(
                        compile(mechanisms_path, recursive=False)
                    ),
                    "target_namespace": target_namespace,
                },
                "objective_names": [
                    "rn_error",
                    "tau_error",
                    "fI_error",
                    "spike_amplitude_error",
                    #"ISI_adaptation_error",
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

        try:
            if feature_selection is None or isinstance(feature_selection, str):
                feature_selection = self.get_best(sort_by=feature_selection)["f"].iloc[
                    -1
                ]
            elif isinstance(feature_selection, int):
                feature_selection = self.get_best()["f"].iloc[feature_selection]
        except IndexError:
            print("Invalid feature selection")
            return plt.figure()

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

        target = mlines.Line2D(
            [], [], color="#1f77b4", marker="s", ls="", label="Target"
        )
        solution = mlines.Line2D(
            [], [], color="#ff6600", marker="o", ls="", label="Solution"
        )

        plt.legend(handles=[target, solution])

        return fig

    def plot_iclamp(
        self,
        cvode=False,
        dt=0.025,
        model_variant="default",
        target_namespace="",
        v_init_config="rest",
        ic_constant=False,
        stim_start=500.0,
        stim_stop=1000.0,
        t_stop=2000.0,
        coreneuron=False,
        stim_amp=0.08,
        passive_features=False,
    ):
        import matplotlib.pyplot as plt
        from neuron import h
        from scipy import optimize
        import logging
        import pprint
        from benchmarks.modeling.utils import ic_constant_f, load_template, run_vclamp
        from benchmarks.modeling.ephys import (
            detect_spikes,
            measure_passive,
            measure_rn_from_vclamp,
        )

        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        def run_iclamp(
            cell,
            amp,
            t0,
            t1,
            dt=0.025,
            record_dt=0.01,
            t_stop=1000.0,
            v_init=-65.0,
            celsius=36,
            use_coreneuron=False,
        ):
            # Create the recording vectors for time and voltage
            vec_t = h.Vector()
            vec_soma_v = h.Vector()
            vec_dend_v = h.Vector()
            vec_dend_ica = h.Vector()
            vec_dend_ik = h.Vector()
            vec_soma_ik = h.Vector()
            vec_soma_ina = h.Vector()
            # vec_soma_ica = h.Vector()
            # vec_soma_cai = h.Vector()
            vec_dend_cai = h.Vector()
            vec_dend_ki = h.Vector()
            vec_soma_ki = h.Vector()
            vec_soma_g_Na = h.Vector()
            vec_soma_g_Kdr = h.Vector()
            vec_soma_g_KCa = h.Vector()
            vec_soma_g_CaN = h.Vector()
            vec_dend_g_KCa = h.Vector()
            vec_dend_g_CaN = h.Vector()
            vec_dend_g_CaL = h.Vector()

            vec_t.record(h._ref_t, record_dt)  # Time
            vec_soma_v.record(cell.soma(0.5)._ref_v, record_dt)  # Voltage
            vec_soma_ik.record(cell.soma(0.5)._ref_ik, record_dt)
            vec_soma_ina.record(cell.soma(0.5)._ref_ina, record_dt)
            # vec_soma_ica.record(cell.soma(0.5)._ref_ica, record_dt)
            # vec_soma_cai.record(cell.soma(0.5)._ref_cai, record_dt)
            vec_soma_ki.record(cell.soma(0.5)._ref_ki, record_dt)

            vec_soma_g_Na.record(cell.soma(0.5)._ref_g_Na_PR, record_dt)
            vec_soma_g_Kdr.record(cell.soma(0.5)._ref_g_K_PR, record_dt)
            # vec_soma_g_KCa.record(cell.soma(0.5)._ref_g_KCa_PR, record_dt)
            # vec_soma_g_CaN.record(cell.soma(0.5)._ref_g_Ca, record_dt)

            vec_dend_v.record(cell.dend(0.5)._ref_v, record_dt)  # Voltage
            vec_dend_cai.record(cell.dend(0.5)._ref_cai, record_dt)
            vec_dend_ki.record(cell.dend(0.5)._ref_ki, record_dt)
            vec_dend_ica.record(cell.dend(0.5)._ref_ica, record_dt)
            vec_dend_ik.record(cell.dend(0.5)._ref_ik, record_dt)

            vec_dend_g_KCa.record(cell.dend(0.5)._ref_g_KCa_PR, record_dt)
            vec_dend_g_CaN.record(cell.dend(0.5)._ref_g_Ca_PR, record_dt)
            # vec_dend_g_CaL.record(cell.dend(0.5)._ref_g_CaL, record_dt)

            # Put an IClamp at the soma
            stim = h.IClamp(0.5, sec=cell.soma)
            stim.delay = t0  # Stimulus stat
            stim.dur = t1 - t0  # Stimulus length
            stim.amp = amp  # strength of current injection

            # Run the Simulation
            h.dt = dt
            h.celsius = celsius
            h.v_init = v_init
            h.init()
            h.finitialize(h.v_init)
            logger.info(f"soma ek = {cell.soma.ek} ena = {cell.soma.ena}")
            logger.info(f"dend ek = {cell.dend.ek} eca = {cell.dend.eca}")
            logger.info(f"soma nao = {cell.soma.nao} ko = {cell.soma.ko}")
            logger.info(f"dend ki = {cell.soma.ki} ko = {cell.dend.ko}")
            logger.info(f"dend cao = {cell.dend.cao} cai = {cell.dend.cai}")

            h.tstop = t_stop
            if use_coreneuron:
                from neuron import coreneuron

                coreneuron.enable = True
            h.run()

            result_dict = {
                "t": np.array(vec_t),
                "soma_v": np.array(vec_soma_v),
                "dend_v": np.array(vec_dend_v),
                "soma_ik": np.array(vec_soma_ik),
                "soma_ki": np.array(vec_soma_ki),
                "soma_ina": np.array(vec_soma_ina),
                "dend_ica": np.array(vec_dend_ica),
                "dend_cai": np.array(vec_dend_cai),
                "dend_ki": np.array(vec_dend_ki),
                "dend_ik": np.array(vec_dend_ik),
                "soma_g_Na": np.array(vec_soma_g_Na),
                "soma_g_Kdr": np.array(vec_soma_g_Kdr),
                # "soma_g_KCa": np.array(vec_soma_g_KCa),
                # "soma_g_CaN": np.array(vec_soma_g_CaN),
                # "dend_g_CaL": np.array(vec_dend_g_CaL),
                "dend_g_CaN": np.array(vec_dend_g_CaN),
                "dend_g_KCa": np.array(vec_dend_g_KCa),
            }

            return result_dict

        # Load the NEURON libraries
        h.load_file("stdrun.hoc")
        h.load_file("rn.hoc")

        # Enable variable time step solver
        h.cvode.use_fast_imem(1)
        h.cvode.cache_efficient(1)
        h.cvode.active(1 if cvode else 0)
        h.secondorder = 2
        h.dt = dt

        # config_dict = None
        # with open(config_path) as f:
        #     config_dict = yaml.load(f, Loader=yaml.FullLoader)
        config_dict = self.config.dopt_params.obj_fun_init_args.protocol_config_dict

        template_dict = config_dict.get("Template", None)
        template_name = None
        template_file = None
        template = None
        if template_dict is None:
            template_name = "MN_nrn"
            template_file = "MN_nrn.hoc"
        else:
            if model_variant in template_dict:
                template_name = template_dict[model_variant]["name"]
                template_file = template_dict[model_variant].get("file", None)
            else:
                raise RuntimeError(f"Unknown model variant {model_variant}")

        from miv_simulator.mechanisms import compile_and_load

        compile_and_load("benchmarks/ca1_pinsky_rinzel_modeling/mechanisms")

        template = load_template(
            template_name,
            os.path.join("benchmarks/ca1_pinsky_rinzel_modeling", template_file),
        )

        target_config = config_dict["Targets"]
        if target_namespace:
            target_config = config_dict["Target namespaces"][target_namespace]
        Rin_config = target_config["Rin"]
        tau0_config = target_config["tau0"]

        # toplevel_param_dict = config_dict.get(toplevel_param_key, None)
        # if toplevel_param_dict is None:
        #     raise RuntimeError(f"Unable to read {toplevel_param_key} configuration")

        # param_dict = toplevel_param_dict.get(param_key, None)
        # if param_dict is None:
        #     param_dict = toplevel_param_dict[int(param_key)]

        best = self.get_best(sort_by="-np.max(y, axis=1)", epsilon="auto")

        param_dict = best["x"].iloc[-1].to_dict()
        param_dict.update(self.config.dopt_params.problem_parameters)

        logger.info(f"{pprint.pformat(param_dict)}")

        if v_init_config == "rest":
            v_init = config_dict["Targets"]["V_rest"]["val"]
        elif v_init_config == "hold":
            v_init = config_dict["Targets"]["V_hold"]["val"]
        else:
            raise RuntimeError(f"Unknown v_init configuration {v_init}")

        cell = template(param_dict)

        h.v_init = v_init
        h.init()
        h.finitialize(h.v_init)
        cell.init_ic(h.v_init)
        ic_constant_0 = cell.soma.ic_constant
        ic_constant_val = ic_constant_0

        # Obtain value for ic_constant such that RMP = v_init
        if ic_constant:
            try:
                x0, res = optimize.brentq(
                    ic_constant_f,
                    -0.1,
                    0.1,
                    args=(template, param_dict, ic_constant_0, h.v_init),
                    xtol=1e-6,
                    maxiter=200,
                    disp=False,
                    full_output=True,
                )
            except ValueError:
                x0 = 0.0
            else:
                if not res.converged:
                    x0 = 0.0
            ic_constant_val = x0 + ic_constant_0
        else:
            if v_init_config == "rest":
                ic_constant_val = best["f"].iloc[-1]["ic_constant_rest"]
            elif v_init_config == "hold":
                ic_constant_val = best["f"].iloc[-1]["ic_constant_hold"]
            else:
                raise RuntimeError(f"Unknown v_init configuration {v_init}")

        cell.soma.ic_constant = ic_constant_val
        h.finitialize(h.v_init)
        h.finitialize(h.v_init)

        h.psection(sec=cell.soma)
        h.psection(sec=cell.dend)

        initial_v_error = ic_constant_f(
            0.0,
            template,
            param_dict,
            cell.soma.ic_constant,
            v_hold=v_init,
            use_cvode=cvode,
        )
        logger.info(f"initial_v_error: {initial_v_error}")
        initial_v_constr = (
            1 if np.isclose(0.0, initial_v_error, rtol=1e-4, atol=1e-4) else 0
        )

        logger.info(f"ic_constant0: {ic_constant_0}")
        logger.info(f"ic_constant: {ic_constant_val}")
        logger.info(f"mean initial vm constraint: {initial_v_constr}")

        if passive_features:
            if "I" in Rin_config:
                Rin_amp = Rin_config["I"][0] * Rin_config.get("I_factor", 1.0)
                iclamp_results = run_iclamp(
                    cell,
                    Rin_amp,
                    stim_start,
                    stim_stop,
                    t_stop=10000.0,
                    v_init=v_init,
                    dt=dt,
                    use_coreneuron=coreneuron,
                )
                t = iclamp_results["t"]
                v = iclamp_results["soma_v"]
                passive_features = measure_passive(t, v, stim_start, stim_stop, Rin_amp)
                Rinp = passive_features["Rinp"]
                tau = passive_features["tau"]

            elif "V" in Rin_config:
                Rin_amp = np.asarray(Rin_config["V"]) * Rin_config.get("V_factor", 1.0)
                Rin_ts = Rin_config["t"]
                vclamp_results = run_vclamp(
                    cell,
                    Rin_amp,
                    Rin_ts,
                    v_init=v_init,
                    dt=dt,
                    use_coreneuron=coreneuron,
                )
                vclamp_results["t0"] = Rin_ts[0]
                vclamp_results["t1"] = Rin_ts[1]
                vclamp_results["t2"] = Rin_ts[2]
                Rinp = measure_rn_from_vclamp(**vclamp_results)
                tau_amp = tau0_config["I"][0] * tau0_config.get("I_factor", 1.0)
                iclamp_results = run_iclamp(
                    cell,
                    tau_amp,
                    stim_start,
                    stim_stop,
                    t_stop=10000.0,
                    v_init=v_init,
                    dt=dt,
                    use_coreneuron=coreneuron,
                )
                t = iclamp_results["t"]
                v = iclamp_results["soma_v"]
                passive_features = measure_passive(t, v, stim_start, stim_stop, tau_amp)
                tau = passive_features["tau"]

            else:
                raise RuntimeError("Unknown configuration for Rinp")

            logger.info(f"soma input resistance: {Rinp} time constant: {tau}")
            # logger.info(f'analytical input and transfer impedance per segment: {pprint.pformat(calcZ(cell))}')

        iclamp_results = run_iclamp(
            cell,
            stim_amp,
            stim_start,
            stim_stop,
            t_stop=t_stop,
            v_init=v_init,
            dt=dt,
            use_coreneuron=coreneuron,
        )
        vec_t = iclamp_results["t"]
        vec_soma_v = iclamp_results["soma_v"]
        vec_dend_v = iclamp_results["dend_v"]
        vec_dend_ik = iclamp_results["dend_ik"]
        vec_soma_ik = iclamp_results["soma_ik"]
        vec_soma_ina = iclamp_results["soma_ina"]
        vec_dend_ik = iclamp_results["dend_ik"]
        vec_dend_ica = iclamp_results["dend_ica"]
        vec_dend_cai = iclamp_results["dend_cai"]
        vec_dend_ki = iclamp_results["dend_ki"]
        vec_soma_ki = iclamp_results["soma_ki"]

        vec_soma_g_Na = iclamp_results["soma_g_Na"]
        vec_soma_g_Kdr = iclamp_results["soma_g_Kdr"]
        # vec_soma_g_KCa = iclamp_results["soma_g_KCa"]
        # vec_soma_g_CaN = iclamp_results["soma_g_CaN"]
        vec_dend_g_KCa = iclamp_results["dend_g_KCa"]
        vec_dend_g_CaN = iclamp_results["dend_g_CaN"]
        # vec_dend_g_CaL = iclamp_results["dend_g_CaL"]

        logger.info(
            f"spikes: {detect_spikes(vec_t, vec_soma_v, stim_start, stim_stop+5.0)}"
        )
        logger.info(f"dend ki min/max: {np.min(vec_dend_ki)} / {np.max(vec_dend_ki)}")
        nrn_type = config_dict["Celltype"]

        nrows = 6
        ncols = 3
        fig, axs = plt.subplots(nrows, ncols)
        axs[0, 0].plot(vec_t, vec_soma_v, linewidth=3, color="r", label="soma_v")
        axs[1, 0].plot(vec_t, vec_dend_v, linewidth=3, color="r", label="dend_v")
        axs[2, 0].plot(vec_t, vec_soma_ina, linewidth=3, color="b", label="soma_ina")
        axs[3, 0].plot(vec_t, vec_dend_ik, linewidth=3, color="b", label="soma_ik")
        axs[4, 0].plot(vec_t, vec_dend_ik, linewidth=3, color="b", label="dend_ik")
        axs[5, 0].plot(vec_t, vec_dend_ica, linewidth=3, color="r", label="dend_ica")
        axs[-1, 0].set_xlabel("Time (ms)")
        axs[0, 0].set_ylabel("V (mV)")

        axs[0, 1].plot(vec_t, vec_soma_g_Na, linewidth=3, color="b", label="soma_g_Na")
        # axs[1, 1].plot(vec_t, vec_soma_g_CaN, linewidth=3, color="b", label="soma_g_CaN")
        axs[2, 1].plot(
            vec_t, vec_soma_g_Kdr, linewidth=3, color="b", label="soma_g_Kdr"
        )
        # axs[3, 1].plot(vec_t, vec_soma_g_KCa, linewidth=3, color="b", label="soma_g_KCa")

        axs[4, 1].plot(
            vec_t, vec_dend_g_CaN, linewidth=3, color="b", label="dend_g_CaN"
        )
        # axs[4, 1].plot(vec_t, vec_dend_g_CaL, linewidth=3, color="g", label="dend_g_CaL")
        axs[5, 1].plot(
            vec_t, vec_dend_g_KCa, linewidth=3, color="b", label="dend_g_KCa"
        )

        axs[0, 2].plot(vec_t, vec_dend_cai, linewidth=3, color="g", label="dend_cai")
        axs[1, 2].plot(vec_t, vec_dend_ki, linewidth=3, color="g", label="dend_ki")
        axs[2, 2].plot(vec_t, vec_soma_ki, linewidth=3, color="g", label="soma_ki")

        for i in range(nrows):
            for j in range(ncols):
                axs[i, j].legend()

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
