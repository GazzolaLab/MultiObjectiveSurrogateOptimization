from machinable import Interface, get


class Baseline(Interface):
    def launch(self):
        get(
            "interface.sopt_modeling",
            [
                lambda _: _.version_from_protocol(
                    "benchmarks/ca1_pinsky_rinzel_modeling/config/CA1_CCKBC.yaml"
                ),
                {
                    "dopt_params": {
                        "n_epochs": 5,
                        "n_initial": 200,
                        "population_size": 400,
                        "num_generations": 200,
                    },
                },
            ],
        ).future()
