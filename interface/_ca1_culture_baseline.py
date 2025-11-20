from machinable import Interface, get


class _Ca1CultureBaseline(Interface):
    def launch(self):
        baseline = [
            """~from_protocol(
                './benchmarks/hippocampal_dissociated_culture/config/optimize_network.yaml',
                302,
                mechanisms_path="benchmarks/hippocampal_dissociated_culture/mechanisms",
                spike_input_path="Slice/CA1_Slice_100_dynamical_response_features_20250912.h5",
                spike_input_namespaces=["Spatiotemporal Feature Spikes drc_features_20250912"],
                spike_input_attr='Spike Train',
                dataset_prefix=".",
                config_prefix="./benchmarks/hippocampal_dissociated_culture/config",
                coordinates_namespace="Generated Coordinates",
                max_walltime_hours=48,
                io_size=1,
                )""",
            {
                "dopt_params": {
                    "optimizer_name": "nsga2",
                    "initial_method": "mc",
                    "n_initial": 10,
                    "initial_maxiter": 10,
                    "n_epochs": 4,
                    "population_size": 200,
                    "num_generations": 400,
                    "termination_conditions": True,
                    "resample_fraction": 0.1,
                    "feasibility_method_name": None,
                    "feasibility_method_kwargs": {},
                    "save": True,
                    "save_surrogate_evals": True,
                }
            },
        ]

        return get(
            "interface.sopt_culture",
            baseline
            + [
                {
                    "dopt_params.surrogate_method_name": "megp",
                    "dopt_params.surrogate_method_kwargs": {},
                },
            ],
        ).launch()
