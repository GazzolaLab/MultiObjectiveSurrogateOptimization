from machinable import Interface, get


class _Ca1Sopt(Interface):

    def populations(self):
        return [
            "SCA",
            "BS",
            "NGFC",
        ]

    def launch(self):
        from models.ops import import_initial_samples

        for nm in self.populations():
            with get("machinable.scope", {"trial": 0}):
                initial = get(
                    "interface.sopt_modeling",
                    [
                        f"""~from_protocol("benchmarks/ca1_pinsky_rinzel_modeling/config/CA1_{nm}.yaml")""",
                        {
                            "dopt_params.surrogate_method_name": "gpr",
                        },
                    ],
                )
                assert initial.cached()
            for trial in range(5):
                with get("machinable.scope", {"trial": trial}):
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
                        e = get(
                            [
                                "interface.sopt_modeling",
                                f"""~from_protocol("benchmarks/ca1_pinsky_rinzel_modeling/config/CA1_{nm}.yaml")""",
                            ],
                            [version]
                            + [
                                {
                                    "dopt_params.n_epochs": 20,
                                }
                            ],
                        ).launch()

                        e.commit()
                        import_initial_samples(
                            file_path=e.output_filepath,
                            source=initial.output_filepath,
                            num=e.num_initial_samples,
                            opt_id=e.config.dopt_params.opt_id,
                            feature_dtypes=e.feature_dtypes,
                            param_names=e.parameter_names,
                        )
