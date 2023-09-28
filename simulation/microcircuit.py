from machinable import get


with get("interface.execution.local"):
    h5_types = get(
        "miv_simulator.interface.h5_types",
        {
            "cell_distributions": {  # "from_file('simulation/config/cell_distributions.yml')",  # {
                "STIM": {"SO": 0, "SP": 10, "SR": 0, "SLM": 0},
                "PYR": {"SO": 0, "SP": 80, "SR": 0, "SLM": 0},
                "PVBC": {"SO": 35, "SP": 10, "SR": 8, "SLM": 0},
                "OLM": {"SO": 44, "SP": 0, "SR": 0, "SLM": 0},
            },
            "synapses": "from_file('simulation/config/synapses.yml')",
        },
    ).launch()

    network = get(
        "miv_simulator.interface.network_architecture",
        {
            "filepath": h5_types.output_filepath,
            "cell_distributions": h5_types.config.cell_distributions,
            "synapses": h5_types.config.synapses,
            "layer_extents": {  # }"from_file('simulation/config/layer_extents.yml')",  # {
                "SO": [[0.0, 0.0, 0.0], [1000.0, 1000.0, 100.0]],
                "SP": [[0.0, 0.0, 100.0], [1000.0, 1000.0, 150.0]],
                "SR": [[0.0, 0.0, 150.0], [1000.0, 1000.0, 350.0]],
                "SLM": [[0.0, 0.0, 350.0], [1000.0, 1000.0, 450.0]],
            },
            "cell_constraints": {
                "PC": {"SP": [100, 120]},
                "PVBC": {"SR": [150, 200]},
            },
        },
    ).launch()

    measure_distances = network.measure_distances().launch()

    synapse_forest = {
        population: network.generate_synapse_forest(
            {
                "population": population,
                "morphology": f"./simulation/morphology/{population}.swc",
            }
        ).launch()
        for population in ["PYR", "PVBC", "OLM"]
    }

    synapses = {
        population: network.distribute_synapses(
            {
                "forest_filepath": synapse_forest[population].output_filepath,
                "cell_types": "from_file('simulation/config/cell_types.yml')",
                "population": population,
                "distribution": "poisson",
                "mechanisms_path": "./simulation/mechanisms",
                "template_path": "./simulation/templates",
                "io_size": 1,
                "write_size": 0,
            }
        ).launch()
        for population in ["PYR", "PVBC", "OLM"]
    }

    connections = {
        population: network.generate_connections(
            {
                "forest_filepath": synapses[population].output_filepath,
                "axon_extents": "from_file('simulation/config/axon_extents.yml')",
                "template_path": "./simulation/templates",
                "io_size": 1,
                "cache_size": 20,
                "write_size": 100,
            }
        ).launch()
        for population in ["PYR", "PVBC", "OLM"]
    }

    graph = get(
        "miv_simulator.interface.neuroh5_graph",
        uses=[
            network,
            *synapse_forest.values(),
            *synapses.values(),
            *connections.values(),
        ],
    ).launch()
