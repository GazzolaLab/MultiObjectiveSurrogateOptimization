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


def init_network_objfun(
    cells: str,
    connections: str,
    cell_types: config.CellTypes,
    synapses: config.Synapses,
    templates: str,
    mechanisms: str,
    use_coreneuron: bool,
    worker=None,
):
    np.seterr(all="raise")
    mechanisms.load(mechanisms)

    # initialize the network
    h = simulator.configure_hoc(coreneuron=use_coreneuron)
    env = simulator.ExecutionEnvironment(comm=worker.merged_comm)
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


def feature_dtypes(experiment):
    return [
        (feature_name, np.float32)
        for feature_name in experiment.config.dopt_params.objective_names
    ]


def network_objfun(parameters, env):
    # - update_network_params
    # - network.run(env, output=False, shutdown=False)
    # - network features and constraints
    pass
