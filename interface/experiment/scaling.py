from machinable import Interface, get


class ScalingExperiment(Interface):

    def launch(self):
        ...
        
    def gh(self, cuda=True):
        for nodes in [1, 2, 4, 8, 16]:
            get(
                "interface.sopt_modeling",
                [
                    "~from_protocol",
                    {
                        "dopt_params": {
                            "surrogate_method_name": "megp",
                            "surrogate_method_kwargs": {"use_cuda": cuda},
                            "n_initial": 500,
                            "population_size": 300,
                        },
                        "nodes": str(nodes),
                        "ranks": 72,
                    },
                ],
            ).launch()

    def gg(self):
        for nodes in [1, 2, 4, 8]:
            get(
                "interface.sopt_modeling",
                [
                    "~from_protocol",
                    {
                        "dopt_params": {
                            "surrogate_method_name": "models.dummy.LR",
                            "surrogate_method_kwargs": {},
                            "n_initial": 500,
                            "population_size": 300,
                        },
                        "nodes": str(nodes),
                        "ranks": 144,
                    },
                ],
            ).launch()