from machinable import Interface, get


class ZeroSampling(Interface):
    def launch(self):
        for nm in ["CCKBC", "PVBC", "SCA"]:
            for trial in range(1):
                with get("machinable.scope", {"trial": trial}):
                    if experiment := get(
                        "interface.sopt_modeling",
                        [
                            f"""~from_protocol(
                                "benchmarks/ca1_pinsky_rinzel_modeling/config/CA1_{nm}.yaml"
                            )""",
                            {
                                "dopt_params": {
                                    "n_epochs": 1,
                                    "n_initial": 2000,
                                },
                            },
                        ],
                    ).future():
                        print(
                            experiment.hash, " finished_at ", experiment.finished_at()
                        )
