import os
from machinable import Interface, get


class _MnSgrad(Interface):
    def launch(self):
        from models.ops import import_initial_samples

        for trial in range(3):
            with get("machinable.scope", {"trial": trial}):
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
                for repeat in range(1):
                    with get(
                        "machinable.scope",
                        {"parent": initial.hash, "repeat": repeat},
                    ):
                        for backbone in ["fttransformer", "resnet"]:
                            vv = []
                            for mode in ["o", "c+o"]:
                                for target in [
                                    "objective constraint",
                                    "objective",
                                    "constraint",
                                ]:
                                    if mode == "o" and "constraint" in target:
                                        continue
                                    # vv.append(
                                    #     {
                                    #         "mode": mode,
                                    #         "feasibility_solving": True,
                                    #         "feasibility_targets": target,
                                    #     }
                                    # )

                                    vv.append(
                                        {
                                            "mode": mode,
                                            "sgrad": True,
                                            "_target": target,
                                        }
                                    )
                            for kwargs in vv:
                                params = {}
                                kwargs["backbone"] = backbone
                                if kwargs.get("sgrad", False):
                                    params = {
                                        "num_generations": 1,
                                        "optimizer_kwargs": dict(
                                            targets=kwargs.pop("_target")
                                        ),
                                    }
                                e = get(
                                    "interface.sopt_modeling",
                                    [
                                        f"""~from_protocol()""",
                                        {
                                            "dopt_params": {
                                                "surrogate_custom_training": "models.ops.joint",
                                                # "n_epochs": 10,
                                                # "resample_fraction": 0.5,
                                                "surrogate_custom_training_kwargs": kwargs,
                                                **params,
                                            }
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
