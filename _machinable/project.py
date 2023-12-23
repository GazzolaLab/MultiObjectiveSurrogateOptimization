from typing import Optional, Tuple, Union
from machinable import Project
from machinable.element import Element


class MultiObjectiveSurrogateOptimization(Project):
    def on_resolve_element(self, module):
        if module == "interface.execution.frontera":
            m, c = super().on_resolve_element("interface.execution.slurm")
            return [
                m,
                {
                    "preamble": '\n\n#export UCX_TLS="knem,dc_x"\n\nibrun',
                },
            ], c

        return super().on_resolve_element(module)

    def on_resolve_remotes(self):
        return {
            "interface.execution.slurm": "url+https://raw.githubusercontent.com/machinable-org/machinable/ea427061c672e8675e2d3b611fb0555bd9d27677/docs/examples/slurm-execution/slurm.py",
            "interface.execution.local": "url+https://raw.githubusercontent.com/machinable-org/machinable/ea427061c672e8675e2d3b611fb0555bd9d27677/docs/examples/mpi-execution/mpi.py",
        }
