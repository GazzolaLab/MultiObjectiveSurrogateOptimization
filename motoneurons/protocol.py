import logging

import numpy as np
import yaml
from neuron import h

import motoneurons.ephys_utils as ephys
from motoneurons.neuron_utils import (
    ic_constant_f,
    run_iclamp,
    run_iclamp_steps,
    run_vclamp,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load(filepath: str):
    with open(filepath) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


class ExperimentalProtocol(object):
    def __init__(self, params_dict, target_namespace):
        self.params_dict = params_dict
        self.target_namespace = target_namespace
        self.init_params(params_dict)

    def init_params(self, params):
        num_config = params["Numerics"]
        self.t0 = num_config["t0"]
        self.tstop = num_config["tstop"]
        self.v_init = num_config["v_init"]
        self.use_cvode = False
        if num_config.get("adaptive", False):
            self.use_cvode = True
        self.use_coreneuron = False
        if num_config.get("use_coreneuron", False):
            self.use_coreneuron = True
        self.dt = num_config.get("dt", 0.01)
        self.record_dt = num_config.get("record_dt", self.dt)

        target_config = self.params_dict["Targets"]
        if self.target_namespace is not None:
            target_namespaces = self.params_dict["Target namespaces"]
            target_config = target_namespaces[self.target_namespace]
        self.v_hold = target_config["V_hold"]["val"]
        self.v_rest = target_config["V_rest"]["val"]

        Rin_config = target_config["Rin"]
        self.target_rn = (Rin_config["lower"][0], Rin_config["upper"][0])
        self.rn_exp_type = "iclamp"
        if "V" in Rin_config:
            self.rn_exp_type = "vclamp"
        tau0_config = target_config["tau0"]
        self.target_tau = (tau0_config["lower"][0], tau0_config["upper"][0])
        f_I_config = target_config["f_I"]
        N_exp = len(f_I_config["I"])
        self.exp_i_inj_amp_f_I = np.asarray(f_I_config["I"]) * f_I_config.get(
            "I_factor", 1.0
        )
        self.exp_i_inj_t0_f_I = f_I_config["t"][0]
        self.exp_i_inj_t1_f_I = f_I_config["t"][1]
        self.exp_i_mean_rate_f_I = (
            np.asarray(f_I_config["mean"]) if "mean" in f_I_config else None
        )
        self.exp_i_ub_rate_f_I = (
            np.asarray(f_I_config["upper"])
            if "upper" in f_I_config
            else self.exp_i_mean_rate_f_I
        )
        self.exp_i_lb_rate_f_I = (
            np.asarray(f_I_config["lower"])
            if "lower" in f_I_config
            else self.exp_i_mean_rate_f_I
        )

        spike_amp_config = target_config["spike_amp"]
        self.exp_i_ub_spk_amp = (
            np.asarray(spike_amp_config["upper"])
            if "upper" in spike_amp_config
            else None
        )
        self.exp_i_lb_spk_amp = (
            np.asarray(spike_amp_config["lower"])
            if "lower" in spike_amp_config
            else None
        )

        spike_adaptation_config = target_config["spike_adaptation"]
        self.exp_i_mean_spk_adaptation = (
            np.asarray(spike_adaptation_config["mean"])
            if "mean" in spike_adaptation_config
            else None
        )
        self.exp_i_ub_spk_adaptation = (
            np.asarray(spike_adaptation_config["upper"])
            if "upper" in spike_adaptation_config
            else self.exp_i_mean_spk_adaptation
        )
        self.exp_i_lb_spk_adaptation = (
            np.asarray(spike_adaptation_config["lower"])
            if "lower" in spike_adaptation_config
            else self.exp_i_mean_spk_adaptation
        )

        self.target_threshold = target_config["threshold"]

        constraint_config = target_config.get("Constraints", {})
        first_ISI_constraints = constraint_config.get("first_ISI", {})
        self.exp_first_ISI_mean = np.asarray(
            first_ISI_constraints.get("mean", np.zeros((N_exp,)))
        )
        print(self.exp_first_ISI_mean)
        self.exp_first_ISI_lower = np.asarray(
            first_ISI_constraints.get("lower", 0.5 * self.exp_first_ISI_mean)
        )
        self.exp_first_ISI_upper = np.asarray(
            first_ISI_constraints.get("upper", 1.5 * self.exp_first_ISI_mean)
        )

    def run_iclamp(self, cell, target, tstop=10000.0):
        target_config = self.params_dict["Targets"]
        if self.target_namespace is not None:
            target_namespaces = self.params_dict["Target namespaces"]
            target_config = target_namespaces[self.target_namespace]

        this_target_config = target_config[target]

        this_target_amp = this_target_config["I"][0] * this_target_config.get(
            "I_factor", 1.0
        )
        tstim = this_target_config.get("t", None)
        if tstim is None:
            t0 = tstop / 2.0
            t1 = t0 + 1000.0
        else:
            t0, t1 = tstim

        t, v = run_iclamp(
            cell, t0=t0, t1=t1, amp=this_target_amp, tstop=tstop, v_init=self.v_hold
        )

        return {"t": t, "v": v, "t0": t0, "t1": t1, "stim_amp": this_target_amp}

    def run_iclamp_steps(self, cell):
        Isteps = np.asarray(
            [
                (
                    amp,
                    self.exp_i_inj_t0_f_I,
                    self.exp_i_inj_t1_f_I,
                )
                for amp in self.exp_i_inj_amp_f_I
            ]
        )
        return run_iclamp_steps(
            cell,
            v_init=self.v_hold,
            Isteps=Isteps,
            record_dt=self.record_dt,
            tstop=self.tstop,
            use_cvode=self.use_cvode,
            use_coreneuron=self.use_coreneuron,
        )

    def run_vclamp(self, cell, target, tstop=10000.0):
        target_config = self.params_dict["Targets"]
        if self.target_namespace is not None:
            target_namespaces = self.params_dict["Target namespaces"]
            target_config = target_namespaces[self.target_namespace]

        this_target_config = target_config[target]

        this_target_amps = np.asarray(this_target_config["V"]) * this_target_config.get(
            "V_factor", 1.0
        )
        tstim = this_target_config.get("t", None)
        if tstim is None:
            t0 = tstop / 2.0
            t1 = t0 + 1000.0
            t2 = t1 + 1000.0
        else:
            t0, t1, t2 = tstim

        vclamp_results = run_vclamp(
            cell,
            ts=[t0, t1, t2],
            amps=this_target_amps,
            t_stop=tstop,
            v_init=self.v_hold,
        )

        return {
            "t0": t0,
            "t1": t1,
            "t2": t2,
            "t": vclamp_results["t"],
            "v": vclamp_results["v"],
            "i": vclamp_results["i"],
        }
