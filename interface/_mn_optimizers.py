from machinable import Interface, get
import os


class _MnOptimizers(Interface):
    def launch(self):
        from models.ops import import_initial_samples

        for trial in range(3):
            with get("machinable.scope", {"trial": trial}):
                for optimizer in ["age", "smpso"]:
                    initial = get(
                        "interface.sopt_modeling",
                        [
                            f"""~from_protocol()""",
                            {
                                "dopt_params.surrogate_method_name": "gpr",
                                "dopt_params.n_epochs": 0,
                            },
                        ],
                    )
                    assert initial.cached()
                    initial.save_attribute("preflight", True)
                    with get("machinable.scope", {"parent": initial.hash}):
                        for version in [
                            {
                                "dopt_params.surrogate_method_name": "gpr",
                            },
                            {
                                "dopt_params.surrogate_method_name": "megp",
                            },
                            "~joint_model(mode='o', backbone='resnet')",
                            "~joint_model(mode='c+o', backbone='resnet')",
                            "~joint_model(mode='o', backbone='fttransformer')",
                            "~joint_model(mode='c+o', backbone='fttransformer')",
                        ]:
                            e = get(
                                "interface.sopt_modeling",
                                [
                                    f"""~from_protocol()""",
                                    version,
                                    {
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
                                    source=initial.output_filepath,
                                    num=e.num_initial_samples,
                                    opt_id=e.config.dopt_params.opt_id,
                                    feature_dtypes=e.feature_dtypes,
                                    param_names=e.parameter_names,
                                )
