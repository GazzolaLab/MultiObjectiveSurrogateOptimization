from machinable import Interface, get


class _Ca1Initial(Interface):
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
        for trial in range(5):
            with get("machinable.scope", {"trial": trial}):
                for nm in self.populations():
                    for version in [
                        {
                            "dopt_params.surrogate_method_name": "gpr",
                        },
                        {
                            "dopt_params.surrogate_method_name": "megp",
                        },
                        "~joint_model(mode='o')",
                        "~joint_model(mode='c+o')",
                    ]:
                        get(
                            [
                                "interface.sopt_modeling",
                                f"""~from_protocol("benchmarks/ca1_pinsky_rinzel_modeling/config/CA1_{nm}.yaml")""",
                            ],
                            version,
                        ).launch()
