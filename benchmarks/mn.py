import os
from neuron import h
import numpy as np
from functools import partial
import logging
from typing import Literal
import sys
from mpi4py import MPI
import time
import importlib
sys.modules['neuron_utils'] = importlib.import_module("benchmarks.motoneuron_modeling.neuron_utils")
sys.modules['ephys_utils'] = importlib.import_module("benchmarks.motoneuron_modeling.ephys_utils")
from benchmarks.motoneuron_modeling.protocol import ExperimentalProtocol
from miv_simulator.mechanisms import load
from scipy import optimize
import benchmarks.motoneuron_modeling.ephys_utils as ephys
from benchmarks.motoneuron_modeling.neuron_utils import (
    ic_constant_f,
    run_iclamp,
    run_iclamp_steps,
    run_vclamp,
    load_template,
)

sys_excepthook = sys.excepthook


def mpi_excepthook(type, value, traceback):
    sys_excepthook(type, value, traceback)
    sys.stdout.flush()
    sys.stderr.flush()
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)


sys.excepthook = mpi_excepthook

SOURCE = os.path.dirname(ephys.__file__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def protocol_obj_fun_init_adapter(protocol_config_dict, template_name, mechanisms, model_variant, target_namespace, worker=None):
    sys.modules['protocol'] = importlib.import_module("benchmarks.motoneuron_modeling.protocol")
    from benchmarks.motoneuron_modeling.dmosopt_MN_nrn import make_obj_fun
    
    load(mechanisms)

    if not hasattr(h, template_name):
        load_template(
            template_name, template_file=f"{SOURCE}/{template_name}.hoc"
        )
    
    return make_obj_fun(
        protocol_config_dict=protocol_config_dict,
        feature_dtypes=feature_dtypes_from_protocol_config(
            protocol_config_dict=protocol_config_dict,
            target_namespace=target_namespace,
        ),
        template_name=template_name,
        target_namespace=target_namespace,
        worker=None
    )

def metadata_from_protocol(optimization):
    protocol_config_dict = optimization.config.dopt_params.obj_fun_init_args.protocol_config_dict
    target_namespace = optimization.config.dopt_params.obj_fun_init_args.target_namespace
    exp_protocol = ExperimentalProtocol(protocol_config_dict,
                                        target_namespace=target_namespace)

    N_exp = len(protocol_config_dict["Targets"]["f_I"]["I"])
    if target_namespace is not None:
        N_exp = len(protocol_config_dict["Target namespaces"][target_namespace]["f_I"]["I"])

    N_spk_amp = min(
        len(exp_protocol.exp_i_lb_spk_amp), len(exp_protocol.exp_i_inj_amp_f_I)
    )
    N_spk_adpt = min(
        len(exp_protocol.exp_i_lb_spk_adaptation), len(exp_protocol.exp_i_inj_amp_f_I)
    )
    obj_targets = {
        "rn": (np.asarray(exp_protocol.target_rn, dtype=np.float32), np.float32, 2),
        "tau": (np.asarray(exp_protocol.target_tau, dtype=np.float32), np.float32, 2),
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
    
    return np.array(
        [tuple((obj_targets[k][0] for k in sorted(obj_targets)))],
        dtype=[
            (f"{k}_target", obj_targets[k][1], obj_targets[k][2])
            for k in sorted(obj_targets)
        ],
    )

def feature_dtypes_from_protocol(optimization):
    return feature_dtypes_from_protocol_config(
        protocol_config_dict = optimization.config.dopt_params.obj_fun_init_args.protocol_config_dict,
        target_namespace = optimization.config.dopt_params.obj_fun_init_args.target_namespace,
    )
    
def feature_dtypes_from_protocol_config(protocol_config_dict, target_namespace):
    N_exp = len(protocol_config_dict["Targets"]["f_I"]["I"])
    if target_namespace is not None:
        N_exp = len(protocol_config_dict["Target namespaces"][target_namespace]["f_I"]["I"])
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
            'initial_v_error_hold',
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
    return feature_dtypes



# --- hard-coded default objective / not used when using ~protocol ----------

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
    ("evaluation_time", np.float32),
]


def init_cell(template_name, pp, v_hold=-60, celsius=36.0, ic_constant_val=None):
    h.cvode.use_fast_imem(1)
    h.cvode.cache_efficient(1)
    h.secondorder = 2

    h.celsius = celsius

    # Create the cell
    template = getattr(h, template_name)
    cell = template(pp)

    # Initialize cell
    h.v_init = v_hold
    h.init()

    if ic_constant_val is None:
        cell.init_ic(h.v_init)
        ic_constant_0 = cell.soma.ic_constant

        # Obtain value for ic_constant such that RMP = v_hold
        x0 = 0.0
        ic_constant0 = ic_constant_0
        try:
            x0, res = optimize.brentq(
                ic_constant_f,
                -0.5,
                0.5,
                args=(template, pp, ic_constant_0, h.v_init),
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

        ic_constant_val = ic_constant_0 + x0

    cell.soma.ic_constant = ic_constant_val
    h.finitialize(h.v_init)
    h.finitialize(h.v_init)

    return cell


def make_obj_fun(**kwargs):
    return partial(obj_fun, **kwargs)


def obj_fun(
    parameters,
    mechanisms: str,
    template_name: str,
    v_hold: float = -60,
    v_rest: float = -57.4,
    rn_exp_type: Literal["iclamp", "vclamp"] = "iclamp",
    worker=None,
):
    start_time = time.time()
    
    load(mechanisms)

    if not hasattr(h, template_name):
        template = load_template(
            template_name, template_file=f"{SOURCE}/{template_name}.hoc"
        )
    else:
        template = getattr(h, template_name)

    cell = init_cell(template_name, parameters, v_hold=v_hold)
    ic_constant_hold = cell.soma.ic_constant

    # Check whether the initial voltage constraint was satisfied
    initial_v_error_hold = float(
        ic_constant_f(0.0, template, parameters, ic_constant_hold, v_hold=v_hold)
    )

    initial_v_constr = 1 if abs(initial_v_error_hold) < 1.0 else -1
    logger.info(f"ic_constant check: {initial_v_error_hold} constr: {initial_v_constr}")

    cell = init_cell(
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
    cell = init_cell(
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
    cell = init_cell(template_name, parameters, v_hold=v_rest)
    ic_constant_rest = cell.soma.ic_constant
    
    evaluation_time = time.time() - start_time

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
            evaluation_time
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
                evaluation_time,
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
