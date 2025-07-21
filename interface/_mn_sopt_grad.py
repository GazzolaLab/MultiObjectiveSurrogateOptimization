from machinable import Interface, get


class _MnSoptGrad(Interface):
    def launch(self):
        for target in [
            "objective",
            "constraint",
            "objective constraint",
        ]:
            for backbone in ["fttransformer", "resnet"]:
                for trial in range(3):
                    with get("machinable.scope", {"trial": trial}):
                        e = get(
                            "interface.sopt_modeling",
                            [
                                f"~from_protocol('benchmarks/motoneuron_modeling/config/wide_ranges.yaml')",
                                {
                                    "dopt_params": {
                                        "dynamic_initial_sampling": "models.ops.dynamic_sampling",
                                        "dynamic_initial_sampling_kwargs": {
                                            "samples_per_iteration": 20,
                                            "max_samples": 1100,  # n_initial
                                            "mode": "c+o",
                                            "backbone": backbone,
                                            "feasibility_solving": 5,
                                            "feasibility_targets": target,
                                            "verbose": 1,
                                        },
                                        "n_epochs": 25,
                                        "n_initial": 50,  # 50 * 11 = 550 = 1/2 of initial
                                    }
                                },
                                f"~joint_model(mode='o', backbone='{backbone}')",
                            ],
                        ).launch()
                        e.save_attribute(
                            "model_name",
                            f"{target}-{backbone}",
                        )
