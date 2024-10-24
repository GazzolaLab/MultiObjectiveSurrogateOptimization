from machinable import Project
import os
import json


class MultiObjectiveSurrogateOptimization(Project):
    def on_resolve_element(self, module):
        if module == "interface.execution.tacc":
            m, c = super().on_resolve_element("interface.execution.slurm")
            return [
                m,
                {"mpi": "ibrun"},
            ], c

        if module == "interface.storage.globus":
            m, c = super().on_resolve_element("interface.storage.globus")
            try:
                with open(os.path.expanduser("~/.globus-config.json"), "r") as fp:
                    version = json.load(fp)
                    for k in ["local_endpoint_directory", "remote_endpoint_directory"]:
                        if k in version:
                            version[k] = os.path.expanduser(
                                os.path.expandvars(
                                    version[k].replace(
                                        "{PROJECT_NAME}", "surrogate-optimization"
                                    )
                                )
                            )
                return [m, version], c
            except FileNotFoundError:
                pass

            return m, c

        return super().on_resolve_element(module)

    def on_resolve_remotes(self):
        version = "main"
        src = f"url+https://raw.githubusercontent.com/machinable-org/machinable/{version}/docs/examples"
        return {
            "interface.storage.globus": src + "/globus-storage/globus.py",
            "interface.execution.slurm": src + "/slurm-execution/slurm.py",
            "interface.execution.mpi": src + "/mpi-execution/mpi.py",
            "interface.execution.require": src + "/require-execution/require.py",
        }
