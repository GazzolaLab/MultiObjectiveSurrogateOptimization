from machinable import Interface, get


class InitialSampling(Interface):
    def launch(self):
        for max_iterations in [3, 5]:
            get(
                "interface.motoneuron",
                [
                    "~from_protocol",
                    "~joint_model",
                    "~dynamic_sampling",
                    {
                        "dopt_params": {
                            "n_epochs": 5,
                            "n_initial": 200,  # num_initial_samples = n_initial * #parameters(11)
                            "population_size": 400,
                            "num_generations": 200,
                            "dynamic_initial_sampling_kwargs": {
                                "max_iterations": max_iterations,
                            },
                        },
                    },
                ],
            ).launch()
