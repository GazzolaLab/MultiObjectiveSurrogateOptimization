import os
import sys

os.environ["DISTWQ_CONTROLLER_RANK"] = "-1"

from miv_simulator import simulator, config
from miv_simulator.mechanisms import load as load_mechanisms
from miv_simulator.optimization import (
    update_network_params,
    network_features,
)
from miv_simulator.synapses import syn_param_from_dict


from functools import partial
import numpy as np
from neuron import h
from mpi4py import MPI


def mpi_excepthook(type, value, traceback):
    sys_excepthook(type, value, traceback)
    sys.stdout.flush()
    sys.stderr.flush()
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)


sys_excepthook = sys.excepthook
sys.excepthook = mpi_excepthook


def feature_dtypes(component):
    return [
        (feature_name, np.float32)
        for feature_name in component.config.dopt_params.objective_names
    ]


def init_network_objfun(
    # network parameters
    cells: str,
    connections: str,
    cell_types: config.CellTypes,
    synapses: config.Synapses,
    templates: str,
    mechanisms: str,
    # simulation parameters
    dt: float,
    v_init: float,
    t_stop: float,
    use_coreneuron: bool,
    # sopt
    opt_targets,
    param_tuples,
    param_names,
    target_populations,
    objective_names,
    worker,
):
    np.seterr(all="raise")
    load_mechanisms(mechanisms)

    simulator.configure_hoc(coreneuron=use_coreneuron)

    env = simulator.ExecutionEnvironment()

    env.load_cells(
        filepath=cells,
        cell_types=cell_types,
        templates=templates,
    )
    env.load_connections(
        filepath=connections,
        cell_filepath=cells,
        synapses=synapses,
    )

    if use_coreneuron:
        h.cvode.cache_efficient(1)
        h.pc.set_maxstep(10)
        h.pc.psolve(0.05)

    param_tuples = [syn_param_from_dict(param_tuple) for param_tuple in param_tuples]

    def from_param_dict(params_dict):
        result = []
        for param_name, param_tuple in zip(param_names, param_tuples):
            result.append((param_tuple, params_dict[param_name]))
        return result

    return partial(
        network_objfun,
        env=env,
        from_param_dict=from_param_dict,
        target_populations=target_populations,
        dt=dt,
        v_init=v_init,
        t_stop=t_stop,
    )


def network_objfun(
    x,
    env,
    from_param_dict,
    target_populations,
    dt,
    v_init,
    t_stop,
):
    param_tuple_values = from_param_dict(x)
    update_network_params(env.adapter(), param_tuple_values)

    env.t_rec.record(h._ref_t)
    env.t_rec.resize(0)
    env.t_vec.resize(0)
    env.id_vec.resize(0)

    t_start = 0.0

    h.t = t_start
    h.secondorder = 2
    h.finitialize(v_init)
    env.pc.timeout(600.0)
    env.pc.psolve(t_stop)

    return network_features(
        env.adapter(),
        target_trj_rate_map_dict={},
        t_start=t_start,
        t_stop=t_stop,
        target_populations=target_populations,
    )
