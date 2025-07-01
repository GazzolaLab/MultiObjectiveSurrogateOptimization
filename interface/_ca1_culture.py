from machinable import Interface, get


class _Ca1Culture(Interface):
    def launch(self):
        get(
            "interface.sopt_culture",
            [
                f"""~from_protocol(
                './benchmarks/hippocampal_dissociated_culture/config/optimize_network.yaml',
                291,
                arena_id="A",
                mechanisms_path="benchmarks/hippocampal_dissociated_culture/mechanisms",
                spike_input_path="Slice/CA1_Slice_100_dynamical_response_features.h5",
                spike_input_namespace="Spatiotemporal Feature Spikes drc_features_20240514",
                spike_input_attr='Spike Train',
                dataset_prefix=".",
                config_prefix="./benchmarks/hippocampal_dissociated_culture/config",
                stimulus_id="Diag",
                coordinates_namespace="Generated Coordinates",
                max_walltime_hours=24,
                io_size=1,
            )""",
                {
                    "dopt_params": {
                        "optimizer_name": "nsga2",
                        "initial_method": "mc",
                        "n_initial": 10,
                        "initial_maxiter": 10,
                        "n_epochs": 10,
                        "population_size": 100,
                        "num_generations": 10,
                        "termination_conditions": True,
                        "resample_fraction": 0.1,
                        "surrogate_method_name": "gpr",
                        "surrogate_method_kwargs": {},
                        "feasibility_method_name": None,
                        "feasibility_method_kwargs": {},
                        "save": True,
                        "save_surrogate_evals": True,
                    }
                },
            ],
        ).launch()

    def test_launch(self):
        get(
            "interface.sopt_culture",
            [
                f"""~from_protocol(
                './benchmarks/hippocampal_dissociated_culture/config/optimize_network_test.yaml',
                279,
                arena_id="A",
                mechanisms_path="benchmarks/hippocampal_dissociated_culture/mechanisms",
                spike_input_path="Slice/CA1_Slice_100_dynamical_response_features.h5",
                spike_input_namespace="Spatiotemporal Feature Spikes drc_features_20240514",
                spike_input_attr='Spike Train',
                dataset_prefix=".",
                config_prefix="./benchmarks/hippocampal_dissociated_culture/config",
                stimulus_id="Diag",
                coordinates_namespace="Generated Coordinates",
                max_walltime_hours="2",
                io_size="1",
            )""",
                {
                    "dopt_params": {
                        "optimizer_name": "nsga2",
                        "initial_method": "mc",  # mc
                        "n_initial": 10,  # 100
                        "initial_maxiter": 10,
                        "n_epochs": 2,  # 2
                        "population_size": 100,
                        "num_generations": 10,
                        "termination_conditions": True,
                        "resample_fraction": 0.1,
                        "surrogate_method_name": "megp",
                        "surrogate_method_kwargs": {},
                        "feasibility_method_name": None,
                        "feasibility_method_kwargs": {},
                        "save": True,
                        "save_surrogate_evals": True,
                    }
                },
            ],
        ).launch()
