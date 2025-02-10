from machinable import Interface, get


class _Ca1Samplers(Interface):
    def populations(self):
        return [
            "SCA",
            "IVY",
            "PVBC",
            "CCKBC",
            "AAC",
            "BS",
            "OLM",
            "NGFC",
            "IS",
        ]

    def launch(self):
        for trial in range(1):
            with get("machinable.scope", {"trial": trial}):
                for nm in self.populations():
                    for sampler in ["glp", "slh", "lh", "mc", "sobol"]:
                        get(
                            "interface.sopt_modeling",
                            [
                                f"""~from_protocol("benchmarks/ca1_pinsky_rinzel_modeling/config/CA1_{nm}.yaml")""",
                                {
                                    "dopt_params.surrogate_method_name": "gpr",
                                    "dopt_params.initial_method": sampler,
                                },
                            ],
                        ).launch()
