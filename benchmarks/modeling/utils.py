import logging

import numpy as np
from neuron import h

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the NEURON libraries
h.load_file("stdrun.hoc")
#h.load_file("rn.hoc")

pc = h.ParallelContext()
if hasattr(pc, "mpiabort_on_error"):
    pc.mpiabort_on_error(0)


def load_template(template_name, template_file=None):
    if template_file is None:
        template_file = f"{template_name}.hoc"
    h.load_file(template_file)
    template = getattr(h, template_name)
    return template


# Analytical input and transfer impedance
def calcZ(cell):
    h.rn(cell)

    sec_dict = {}
    for sec in cell.all:
        seg_dict = {}
        for seg in sec.allseg():
            seg_dict[seg.x] = {
                "distance": h.distance(seg.x, sec=sec),
                "input": h.zz.input(seg.x, sec=sec),
                "transfer": h.zz.transfer(seg.x, sec=sec),
            }
        sec_dict[sec.name()] = seg_dict

    return sec_dict


def ic_constant_f(
    x,
    template,
    param_dict,
    ic_constant,
    v_hold=-60,
    tstop=1000.0,
    dt=0.01,
    record_dt=0.01,
    celsius=36.0,
    use_cvode=False,
    use_coreneuron=True,
):
    h.cvode.use_fast_imem(1)
    h.cvode.cache_efficient(1)
    h.secondorder = 2
    h.dt = dt

    if record_dt < dt:
        record_dt = dt

    # Enable variable time step solver
    if use_cvode:
        h.cvode.active(1)

    if use_coreneuron:
        from neuron import coreneuron

        coreneuron.enable = True

    h.celsius = celsius

    cell = template(param_dict)

    # Create the recording vectors for time and voltage
    vec_t = h.Vector()
    vec_v = h.Vector()
    vec_t.record(h._ref_t, record_dt)  # Time
    vec_v.record(cell.soma(0.5)._ref_v, record_dt)  # Voltage

    # Run the simulation
    h.tstop = tstop
    h.v_init = v_hold
    h.init()
    cell.soma.ic_constant = ic_constant + round(x, 6)
    h.finitialize(h.v_init)
    h.finitialize(h.v_init)
    try:
        h.run()
    except:
        pass

    t = vec_t.as_numpy()
    v = vec_v.as_numpy()
    mean_v = np.mean(v) if np.max(v) < 0.0 else 0.0

    return mean_v - v_hold


def run_iclamp(
    cell,
    amp,
    t0,
    t1,
    v_init=-65,
    tstop=1000.0,
    dt=0.01,
    record_dt=0.01,
    celsius=36.0,
    use_cvode=False,
    use_coreneuron=True,
):
    h.cvode.use_fast_imem(1)
    h.cvode.cache_efficient(1)
    h.secondorder = 2
    h.dt = dt

    if record_dt < dt:
        record_dt = dt

    # Enable variable time step solver
    if use_cvode:
        h.cvode.active(1)

    if use_coreneuron:
        from neuron import coreneuron

        coreneuron.enable = True

    h.celsius = celsius

    # Create the recording vectors for time and voltage
    vec_t = h.Vector()
    vec_v = h.Vector()
    vec_t.record(h._ref_t, record_dt)  # Time
    vec_v.record(cell.soma(0.5)._ref_v, record_dt)  # Voltage

    # Put an IClamp at the soma
    stim = h.IClamp(0.5, sec=cell.soma)
    stim.delay = t0  # Stimulus start
    stim.dur = t1 - t0  # Stimulus length
    stim.amp = amp  # strength of current injection

    if tstop < t1 + 1.0:
        tstop = t1 + 1.0

    # Run the Simulation
    h.tstop = tstop
    h.v_init = v_init
    h.init()
    h.finitialize(h.v_init)
    h.run()

    return np.array(vec_t), np.array(vec_v)


def run_iclamp_steps(
    cell,
    Isteps=[(0, 500, 1000), (-0.1, 500, 1000), (0.1, 500, 1000), (0.5, 500, 1000)],
    **kwargs,
):
    results = []
    for amp, t0, t1 in Isteps:
        try:
            t, v = run_iclamp(cell, amp, t0, t1, **kwargs)
        except:
            results.append(None)
        else:
            results.append({"t": t, "v": v})

    return results


def run_vclamp(
    cell,
    amps,
    ts,
    mechanism_names=[],
    rs=0.01,
    dt=0.025,
    record_dt=0.01,
    t_stop=None,
    v_init=-65.0,
    celsius=36,
    use_coreneuron=False,
):
    # Create the recording vectors for time and voltage
    vec_t = h.Vector()
    vec_v = h.Vector()
    vec_i = h.Vector()

    sec = cell.soma

    vec_t.record(h._ref_t, record_dt)  # Time
    vec_v.record(sec(0.5)._ref_v, record_dt)  # Voltage
    vec_i_dict = {}
    for mechanism in mechanism_names:
        if hasattr(sec(0.5), f"_ref_i{mechanism}"):
            vec_i = h.Vector()
            vec_i.record(getattr(sec(0.5), f"_ref_i{mechanism}"), record_dt)
            vec_i_dict[mechanism] = vec_i

    c = h.SEClamp(sec(0.5))
    c.dur1 = ts[0]
    c.dur2 = ts[1] - ts[0]
    c.dur3 = ts[2] - ts[1]
    c.amp1 = amps[0]
    c.amp2 = amps[1]
    c.amp3 = amps[2]
    c.rs = rs
    vec_i.record(c._ref_i, record_dt)  # Voltage clamp current

    # Run the Simulation
    h.dt = dt
    h.celsius = celsius
    h.v_init = v_init
    h.init()
    h.finitialize(h.v_init)

    if t_stop is None:
        t_stop = ts[2]
    h.tstop = t_stop
    if use_coreneuron:
        from neuron import coreneuron

        coreneuron.enable = True

    h.run()

    results = {
        "t": np.array(vec_t),
        "i": np.array(vec_i),
        "v": np.array(vec_v),
    }

    for mechanism in mechanism_names:
        results[f"i_{mechanism}"] = np.array(vec_i_dict[mechanism])

    return results
