import os
from machinable import Interface, get


class _Ca1Sgrad(Interface):
    def populations(self):
        return [
            # "SCA",
            "BS",
            # "NGFC",
        ]

    def launch(self):
        from models.ops import import_initial_samples

        for nm in self.populations():
            for trial in range(1):
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
                    )
                    assert initial.cached()
                    initial.save_attribute("preflight", True)
                    with get("machinable.scope", {"parent": initial.hash}):
                        for fmi in [1, 3, 5, 10, 50]:
                            for fl in ["", "+var"]:
                                e = get(
                                    "interface.sopt_modeling",
                                    [
                                        f"""~from_protocol("benchmarks/ca1_pinsky_rinzel_modeling/config/CA1_{nm}.yaml")""",
                                        {
                                            "dopt_params": {
                                                "surrogate_custom_training": "models.ops.joint",
                                                "surrogate_custom_training_kwargs": {
                                                    "feasibility_solving": True,
                                                    "feasibility_max_iterations": fmi,
                                                    "feasibility_loss": fl,
                                                },
                                                "n_epochs": 25,
                                            }
                                        },
                                    ],
                                ).launch()

                                if not os.path.isfile(e.output_filepath):
                                    e.commit()
                                    import_initial_samples(
                                        file_path=e.output_filepath,
                                        source=initial.output_filepath,
                                        num=e.num_initial_samples,
                                        opt_id=e.config.dopt_params.opt_id,
                                        feature_dtypes=e.feature_dtypes,
                                        param_names=e.parameter_names,
                                    )
