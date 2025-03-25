from machinable import Interface, get


class _Ca1Bs(Interface):
    def launch(self):
        nm = "BS"
        for repeat in range(3):
            with get(
                "machinable.scope",
                {"trial": 0, "repeat": repeat},
            ):
                for backbone in [
                    "fttransformer",
                    # "resnet"
                ]:
                    for target in [
                        "objective constraint distance",
                        "objective distance",
                        "objective constraint",
                        "objective",
                        "constraint",
                        "distance",
                    ]:
                        get(
                            "interface.sopt_modeling",
                            [
                                f"""~from_protocol("benchmarks/ca1_pinsky_rinzel_modeling/config/CA1_{nm}.yaml")""",
                                {
                                    "dopt_params": {
                                        "dynamic_initial_sampling": "models.ops.dynamic_sampling",
                                        "dynamic_initial_sampling_kwargs": {
                                            "samples_per_iteration": 100,
                                            "max_samples": 1200,
                                            "stop_condition": False,
                                            "optimizer_sampling": None,
                                            "feasibility_solving": True,
                                            "feasibility_targets": target,
                                            "backbone": backbone,
                                        },
                                        "n_initial": 10,
                                        "surrogate_custom_training": "models.ops.joint",
                                        "n_epochs": 25,
                                        "surrogate_custom_training_kwargs": {
                                            "mode": "c+o",
                                            "sensitivity": True,
                                            "backbone": backbone,
                                        },
                                    }
                                },
                            ],
                        ).launch()
