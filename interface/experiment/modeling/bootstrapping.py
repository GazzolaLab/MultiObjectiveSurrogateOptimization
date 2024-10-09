from machinable import Interface, get


class InitialRange(Interface):
    def launch(self):
        get(
            "interface.sopt_modeling",
            [
                "~from_protocol('benchmarks/motoneuron_modeling/config/motoneuron.yaml')",
                "~dynamic_sampling",
            ],
        ).launch()

        return self
