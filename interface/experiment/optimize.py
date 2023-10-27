from machinable import Component
from mpi4py import MPI


class Optimize(Component):
    def on_write_meta_data(self):
        return MPI.COMM_WORLD.Get_rank() == 0
