from machinable import Interface, get
import os


class _Ca1Optimizers(Interface):
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
        from models.ops import import_initial_samples

        for trial in range(3):
            with get("machinable.scope", {"trial": trial}):
                for nm in self.populations():
                    for optimizer in ["age", "smpso"]:
                        nsga2 = get(
                            "interface.sopt_modeling",
                            [
                                f"""~from_protocol("benchmarks/ca1_pinsky_rinzel_modeling/config/CA1_{nm}.yaml")""",
                                {
                                    "dopt_params.surrogate_method_name": "gpr",
                                    "dopt_params.initial_method": "slh",
                                },
                            ],
                        )
                        assert nsga2.cached()
                        nsga2.save_attribute("preflight", True)

                        e = get(
                            "interface.sopt_modeling",
                            [
                                f"""~from_protocol("benchmarks/ca1_pinsky_rinzel_modeling/config/CA1_{nm}.yaml")""",
                                {
                                    "dopt_params.surrogate_method_name": "gpr",
                                    "dopt_params.initial_method": "slh",
                                    "dopt_params.optimizer_name": optimizer,
                                },
                            ],
                        ).launch()

                        if os.environ.get("LAUNCH", 0) and not os.path.isfile(
                            e.output_filepath
                        ):
                            e.commit()
                            import_initial_samples(
                                file_path=e.output_filepath,
                                source=nsga2.output_filepath,
                                num=e.num_initial_samples,
                                opt_id=e.config.dopt_params.opt_id,
                                feature_dtypes=e.feature_dtypes,
                                param_names=e.parameter_names,
                            )
