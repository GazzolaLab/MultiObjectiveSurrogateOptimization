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
                        vv = []
                        for mode in ["o", "c+o"]:
                            for target in [
                                "objective distance",
                                "objective",
                                "distance",
                            ]:
                                vv.append(
                                    {
                                        "mode": mode,
                                        "feasibility_solving": True,
                                        "feasibility_targets": target,
                                    }
                                )

                                vv.append(
                                    {"mode": mode, "sgrad": True, "_target": target}
                                )

                        for kwargs in vv:
                            params = {}
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
                                    f"""~from_protocol("benchmarks/ca1_pinsky_rinzel_modeling/config/CA1_{nm}.yaml")""",
                                    {
                                        "dopt_params": {
                                            "surrogate_custom_training": "models.ops.joint",
                                            "n_epochs": 75,
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
