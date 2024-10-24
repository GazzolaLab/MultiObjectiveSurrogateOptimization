from machinable import Interface, get


class ScalingExperiment(Interface):

    def launch(self):
        for nodes in [1, 2, 4]:
            get(
                "interface.sopt_modeling",
                [
                    "~from_protocol",
                    {
                        "dopt_params": {
                            "surrogate_method_name": "megp",
                            "surrogate_method_kwargs": {"use_cuda": True},
                            "n_initial": 500,
                            "population_size": 300,
                        },
                        "nodes": str(nodes),
                        "ranks": 72,
                    },
                ],
            ).launch()
