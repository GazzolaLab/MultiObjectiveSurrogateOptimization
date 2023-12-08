import os
from neuron import h
import numpy as np
from functools import partial
import logging
from typing import Literal
import ephys_utils as ephys
from miv_simulator.mechanisms import compile_and_load

try:
    import dmosopt_MN_nrn
except ImportError as _ex:
    raise ModuleNotFoundError(
        "Make sure the Motoneuron-Modeling repo is in your PATH!"
    ) from _ex

SOURCE = os.path.dirname(dmosopt_MN_nrn.__file__)

from neuron_utils import (
    ic_constant_f,
    run_iclamp,
    run_iclamp_steps,
    run_vclamp,
    load_template,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


def make_obj_fun(template_name, **kwargs):
    return partial(obj_fun, template_name=template_name, **kwargs)


def obj_fun(
    parameters,
    template_name: str = "rn",
    v_hold: float = -60,
    v_rest: float = -57.4,
    rn_exp_type: Literal["iclamp", "vclamp"] = "iclamp",
    worker=None,
):
    if worker:
        print("Worker: {worker}")

    compile_and_load(f"{SOURCE}/mechanisms")

    load_template(template_name, template_file=f"{SOURCE}/{template_name}.hoc")

    template = getattr(h, template_name)

    cell = dmosopt_MN_nrn.init_cell(template_name, parameters, v_hold=v_hold)
    ic_constant_hold = cell.soma.ic_constant

    # Check whether the initial voltage constraint was satisfied
    initial_v_error_hold = float(
        ic_constant_f(0.0, template, parameters, ic_constant_hold, v_hold=v_hold)
    )

    initial_v_constr = 1 if abs(initial_v_error_hold) < 1.0 else -1
    logger.info(f"ic_constant check: {initial_v_error_hold} constr: {initial_v_constr}")

    cell = dmosopt_MN_nrn.init_cell(
        template_name, parameters, v_hold=v_hold, ic_constant_val=ic_constant_hold
    )

    rn, tau = np.nan, np.nan
    iclamp_results = None
    vclamp_results = None

    # Measure input resistance and membrane time constant
    if initial_v_constr > 0:
        # Run single current injection to measure subthreshold features
        try:
            if rn_exp_type == "iclamp":
                # Rin
                this_target_amp = -100.0 * 1.0e-3
                t0 = 250
                t1 = 1250
                tstop = 3000.0
                t, v = run_iclamp(
                    cell, t0=t0, t1=t1, amp=this_target_amp, tstop=tstop, v_init=v_hold
                )
                iclamp_results = {
                    "t": t,
                    "v": v,
                    "t0": t0,
                    "t1": t1,
                    "stim_amp": this_target_amp,
                }
            elif rn_exp_type == "vclamp":
                # Rin
                this_target_amps = np.asarray(-60.0) * 1.0
                tstop = 10000.0
                t0 = tstop / 2.0
                t1 = t0 + 1000.0
                t2 = t1 + 1000.0
                vclamp_results = run_vclamp(
                    cell,
                    ts=[t0, t1, t2],
                    amps=this_target_amps,
                    t_stop=tstop,
                    v_init=v_hold,
                )
                vclamp_results = {
                    "t0": t0,
                    "t1": t1,
                    "t2": t2,
                    "t": vclamp_results["t"],
                    "v": vclamp_results["v"],
                    "i": vclamp_results["i"],
                }

                # tau0
                this_target_amp = -0.1
                t0 = 250
                t1 = 1250
                tstop = 3000.0
                t, v = run_iclamp(
                    cell, t0=t0, t1=t1, amp=this_target_amp, tstop=tstop, v_init=v_hold
                )
                iclamp_results = {
                    "t": t,
                    "v": v,
                    "t0": t0,
                    "t1": t1,
                    "stim_amp": this_target_amp,
                }
        except:
            pass
        else:
            passive_results = ephys.measure_passive(**iclamp_results)
            rn = passive_results["Rinp"]
            tau = passive_results["tau"]
            if vclamp_results is not None:
                rn = ephys.measure_rn_from_vclamp(**vclamp_results)

    target_rn = (540.0, 598.0)
    target_tau = (16.308, 19.375)
    rn_obj_value = range_distance(rn, target_rn[0], target_rn[1]) ** 2
    tau_obj_value = range_distance(tau, target_tau[0], target_tau[1]) ** 2

    tau_constr = 1 if ((tau > 0.0) and (tau < 1000.0)) else -1
    rn_constr = 1 if ((rn > 0.0) and (rn < 1000.0)) else -1

    exp_i_inj_t0_f_I = 500
    exp_i_inj_t1_f_I = 1500
    exp_i_inj_amp_f_I = np.asarray([20, 30, 40, 50, 60, 70, 80]) * 1.0e-3

    # Run iclamp experiments
    iclamp_results = []
    cell = dmosopt_MN_nrn.init_cell(
        template_name, parameters, v_hold=v_hold, ic_constant_val=ic_constant_hold
    )
    iclamp_results = run_iclamp_steps(
        cell,
        v_init=v_hold,
        Isteps=np.asarray(
            [
                (
                    amp,
                    exp_i_inj_t0_f_I,
                    exp_i_inj_t1_f_I,
                )
                for amp in exp_i_inj_amp_f_I
            ]
        ),
        record_dt=0.01,
        tstop=2000,
        use_cvode=False,
        use_coreneuron=True,
    )

    # Measure spike features
    (
        pre_spk_cnt,
        spk_cnt,
        spk_infos,
        thresholds,
        mean_spike_amplitudes,
    ) = ephys.measure_spike_features(
        iclamp_results, exp_i_inj_t0_f_I, exp_i_inj_t1_f_I + 2.0
    )

    pre_spk_count_constr = -1 if np.sum(pre_spk_cnt) > 0 else 1

    ISI_values = ephys.measure_ISI(exp_i_inj_amp_f_I, spk_infos)

    exp_i_lb_spk_adaptation = np.asarray([1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2])
    exp_i_ub_spk_adaptation = np.asarray([1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6])

    ISI_adaptation_dists = list(
        map(
            lambda ratio, target_range: range_distance(
                ratio * 100.0, target_range[0] * 100.0, target_range[1] * 100.0
            ),
            ISI_values["ratio"],
            zip(
                exp_i_lb_spk_adaptation,
                exp_i_ub_spk_adaptation,
            ),
        )
    )
    ISI_adaptation_obj_value = np.mean([dist**2 for dist in ISI_adaptation_dists])
    ISI_adaptation_constr = -1 if np.isnan(ISI_adaptation_obj_value) else 1

    exp_first_ISI_lower = np.zeros(len([20, 30, 40, 50, 60, 70, 80]))

    first_ISI_constr = 1 if np.all(ISI_values["first"] > exp_first_ISI_lower) else -1

    fI_values = ephys.measure_fI(
        spk_cnt,
        exp_i_inj_t0_f_I,
        exp_i_inj_t1_f_I,
        exp_i_inj_amp_f_I,
    )

    exp_i_mean_rate_f_I = [3.88, 9.09, 11.75, 14.29, 15.96, 16.58, 18.25]

    fI_mean_target_rate_diff = np.mean(
        [
            (target_rate - rate) ** 2
            for rate, target_rate in zip(fI_values["frequency"], exp_i_mean_rate_f_I)
        ]
    )

    exp_i_lb_rate_f_I = exp_i_mean_rate_f_I
    exp_i_ub_rate_f_I = exp_i_mean_rate_f_I
    fI_range_dists = list(
        map(
            lambda rate, target_range: range_distance(
                rate, target_range[0], target_range[1]
            ),
            fI_values["frequency"],
            zip(exp_i_lb_rate_f_I, exp_i_ub_rate_f_I),
        )
    )
    fI_obj_value = np.mean([dist**2 for dist in fI_range_dists])

    # Compute objectives
    exp_i_ub_spk_amp = [80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0]
    exp_i_lb_spk_amp = [60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0]
    mean_spike_amplitude_range_dists = list(
        map(
            lambda amp, target_amp: None
            if np.isnan(target_amp[0])
            else range_distance(amp, target_amp[0], target_amp[1]),
            mean_spike_amplitudes,
            zip(exp_i_lb_spk_amp, exp_i_ub_spk_amp),
        )
    )
    mean_spike_amplitude_obj_value = np.mean(
        [
            dist**2
            for dist in filter(
                lambda x: False if x is None else True, mean_spike_amplitude_range_dists
            )
        ]
    )
    spike_amplitude_constr = -1 if np.isnan(mean_spike_amplitude_obj_value) else 1

    # Check for fI monotonicity
    fI_rate_diff = np.diff(fI_values["frequency"][:-1])
    monotonic_fI_constr = 1 if np.all(fI_rate_diff > 0) else -1

    # Obtain ic_constant for v_rest target
    cell = dmosopt_MN_nrn.init_cell(template_name, parameters, v_hold=v_rest)
    ic_constant_rest = cell.soma.ic_constant

    # Pass to dmosopt
    feature_list = [
        (
            ic_constant_hold,
            ic_constant_rest,
            initial_v_error_hold,
            rn,
            tau,
            fI_values,
            fI_mean_target_rate_diff,
            ISI_values,
            thresholds,
            mean_spike_amplitudes,
        )
    ]

    feature_values = np.asarray(
        [
            (
                ic_constant_hold,
                ic_constant_rest,
                initial_v_error_hold,
                rn,
                tau,
                fI_values,
                fI_mean_target_rate_diff,
                ISI_values,
                thresholds,
                mean_spike_amplitudes,
            )
        ],
        dtype=np.dtype(feature_dtypes),
    )

    obj_values = np.asarray(
        [
            rn_obj_value,
            tau_obj_value,
            fI_obj_value,
            mean_spike_amplitude_obj_value,
            ISI_adaptation_obj_value,
        ],
        dtype=np.float32,
    )
    constr_values = np.asarray(
        [
            monotonic_fI_constr,
            rn_constr,
            tau_constr,
            spike_amplitude_constr,
            first_ISI_constr,
            ISI_adaptation_constr,
            pre_spk_count_constr,
            initial_v_constr,
        ],
        dtype=np.float32,
    )

    return obj_values, feature_values, constr_values


def range_distance(x, lb, ub):
    # Returns 0. if x is within the range [lb, ub], otherwise returns the smaller of the distance between x and lb, ub
    return 0.0 if (x >= lb) and (x <= ub) else min(abs(x - lb), abs(x - ub))
