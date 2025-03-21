import os
from machinable import Interface, get


class _Ca1Sa(Interface):
    def populations(self):
        return [
            "SCA",
            "BS",
            "NGFC",
            "OLM",
        ]

    def launch(self):
        from models.ops import import_initial_samples

        for nm in self.populations():
            for trial in range(5):
                with get("machinable.scope", {"trial": trial}):
                    initial = get(
                        "interface.sopt_modeling",
                        [
                            f"""~from_protocol("benchmarks/ca1_pinsky_rinzel_modeling/config/CA1_{nm}.yaml")""",
                            {
                                "dopt_params.surrogate_method_name": "gpr",
                                "dopt_params.n_epochs": 0,
                            },
                        ],
                    ).launch()
                    initial.save_attribute("preflight", True)
                    if initial.cached():
                        for version in [
                            [
                                "~joint_model(mode='c+o', backbone='fttransformer')",
                                {"dopt_params.sensitivity_method_name": "dgsm"},
                            ],
                            [
                                "~joint_model(mode='c+o', backbone='fttransformer')",
                                {"dopt_params.sensitivity_method_name": "fast"},
                            ],
                            "~joint_model(mode='c+o', backbone='fttransformer', sensitivity=True)",
                            "~joint_model(mode='c+o', backbone='fttransformer', sensitivity='cross_check')",
                        ]:
                            with get(
                                "machinable.scope",
                                {"parent": initial.hash, "renorm": "max"},
                            ):
                                e = get(
                                    [
                                        "interface.sopt_modeling",
                                        f"""~from_protocol("benchmarks/ca1_pinsky_rinzel_modeling/config/CA1_{nm}.yaml")""",
                                    ],
                                    version,
                                ).launch()

                                if os.environ.get("LAUNCH", 0) and not os.path.isfile(
                                    e.output_filepath
                                ):
                                    e.commit()
                                    import_initial_samples(
                                        file_path=e.output_filepath,
                                        source=initial.output_filepath,
                                        num=e.num_initial_samples,
                                        opt_id=e.config.dopt_params.opt_id,
                                        feature_dtypes=e.feature_dtypes,
                                        param_names=e.parameter_names,
                                    )
