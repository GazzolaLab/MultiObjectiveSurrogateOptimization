from machinable import Interface, get


class _MnRandom(Interface):

    def launch(self):
        for trial in range(3):
            with get("machinable.scope", {"trial": trial}):
                for c in [
                    "",
                    "'benchmarks/motoneuron_modeling/config/wide_ranges.yaml'",
                ]:
                    for sampler in ["slh", "lh", "mc", "sobol"]:
                        get(
                            "interface.sopt_modeling",
                            [
                                f"""~from_protocol({c})""",
                                {
                                    "dopt_params.surrogate_method_name": "gpr",
                                    "dopt_params.n_epochs": 0,
                                    "dopt_params.initial_method": sampler,
                                    "dopt_params.n_initial": 319,  # * 11 = 3509
                                },
                            ],
                        ).launch()
