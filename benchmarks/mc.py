import os
import sys

os.environ["DISTWQ_CONTROLLER_RANK"] = "-1"

from miv_simulator import simulator, config

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


def init_controller(subworld_size, use_coreneuron):
    h.nrnmpi_init()
    h("objref pc, cvode")
    h.cvode = h.CVode()
    h.pc = h.ParallelContext()
    h.pc.subworlds(subworld_size)
    if use_coreneuron:
        from neuron import coreneuron

        coreneuron.enable = True
        coreneuron.verbose = 0
        h.cvode.cache_efficient(1)
        h.finitialize(-65)
        h.pc.set_maxstep(10)
        h.pc.psolve(0.05)


def init_network_objfun(
    cells: str,
    connections: str,
    cell_types: config.CellTypes,
    synapses: config.Synapses,
    templates: str,
    mechanisms: str,
    use_coreneuron: bool,
):
    np.seterr(all="raise")
    mechanisms.load(mechanisms)

    # initialize the network
    h = simulator.configure_hoc(coreneuron=use_coreneuron)
    env = simulator.ExecutionEnvironment()  # seed=self.seed)
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

    return partial(network_objfun, env=env)


def network_objfun(parameters, env):
    # - update_network_params
    # - network.run(env, output=False, shutdown=False)
    # - network features and constraints
    pass
