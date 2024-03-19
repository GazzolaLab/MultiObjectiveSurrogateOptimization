from machinable import Project
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
        return {
            "interface.storage.globus": "url+https://raw.githubusercontent.com/machinable-org/machinable/2670e9626eb548f6ce2301923be1f49642086d8c/docs/examples/globus-storage/globus.py",
            "interface.execution.slurm": "url+https://raw.githubusercontent.com/machinable-org/machinable/2670e9626eb548f6ce2301923be1f49642086d8c/docs/examples/slurm-execution/slurm.py",
            "interface.execution.local": "url+https://raw.githubusercontent.com/machinable-org/machinable/2670e9626eb548f6ce2301923be1f49642086d8c/docs/examples/mpi-execution/mpi.py",
        }
