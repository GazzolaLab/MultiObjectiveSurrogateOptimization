import os
from machinable import Interface, get


class _MnSoptGrad(Interface):

    def launch(self):
        for trial in range(3):
            with get("machinable.scope", {"trial": trial}):
                for backbone in ["fttransformer", "resnet"]:
                    for target in [
                        "objective constraint",
                        "objective",
                        "constraint",
                    ]:
                        get(
                            "interface.sopt_modeling",
                            [
                                """~from_protocol('benchmarks/motoneuron_modeling/config/wide_ranges.yaml')""",
                                {
                                    "dopt_params": {
                                        "dynamic_initial_sampling": "models.ops.dynamic_sampling",
                                        "dynamic_initial_sampling_kwargs": {
                                            "samples_per_iteration": 100,
                                            "max_samples": 1100,
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
