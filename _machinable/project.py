from typing import Optional, Tuple, Union
from machinable import Project
from machinable.element import Element
import os
import json

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
            
        if module == "interface.storage.globus":
            m, c = super().on_resolve_element("interface.storage.globus")
            try:
                with open(os.path.expanduser("~/.globus-config.json"), 'r') as fp:
                    version = json.load(fp)
                return [m,version], c
            except FileNotFoundError:
                pass
            
            return m, c

        return super().on_resolve_element(module)

    def on_resolve_remotes(self):
        return {
            "interface.storage.globus": "url+https://raw.githubusercontent.com/machinable-org/machinable/86045f0da6ef1a53b3d2421b44f878be380a54e5/docs/examples/globus-storage/globus.py",
            "interface.execution.slurm": "url+https://raw.githubusercontent.com/machinable-org/machinable/86045f0da6ef1a53b3d2421b44f878be380a54e5/docs/examples/slurm-execution/slurm.py",
            "interface.execution.local": "url+https://raw.githubusercontent.com/machinable-org/machinable/86045f0da6ef1a53b3d2421b44f878be380a54e5/docs/examples/mpi-execution/mpi.py",
        }
