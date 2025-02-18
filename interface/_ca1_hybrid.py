import os
from machinable import Interface, get


class _Ca1Hybrid(Interface):
    def populations(self):
        return ["BS"]

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
                    ).launch()
                    initial.save_attribute("preflight", True)
                    if initial.cached():
                        vv = [{}]
                        for mode in ["o", "c+o"]:
                            for target in [
                                "objective distance",
                                "objective",
                                "-objective",
                                "-objective distance",
                                "distance",
                            ]:
                                vv.append(
                                    {
                                        "dopt_params": {
                                            "surrogate_custom_training": "models.ops.joint",
                                            "surrogate_custom_training_kwargs": {
                                                "mode": mode,
                                                "objectives": False,
                                                "feasibility_solving": True,
                                                "feasibility_targets": target,
                                            },
                                        }
                                    }
                                )
                        for version in vv:
                            with get("machinable.scope", {"parent": initial.hash}):
                                e = get(
                                    [
                                        "interface.sopt_modeling",
                                        f"""~from_protocol("benchmarks/ca1_pinsky_rinzel_modeling/config/CA1_{nm}.yaml")""",
                                    ],
                                    [version]
                                    + [
                                        {
                                            "dopt_params.n_epochs": 10,
                                        }
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
