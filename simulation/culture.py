from machinable import get
from miv_simulator.utils import from_yaml
from typing import Literal


def motoneurons(num_channels: Literal[64, 128, 256] = 64):
    h5_types = get(
        "miv_simulator.interface.h5_types",
        [
            {
                "cell_distributions": {
                    "STIM": {"SO": 0, "SP": num_channels, "SR": 0, "SLM": 0},
                    "PYR": {"SO": 0, "SP": 223, "SR": 0, "SLM": 0},
                    "PVBC": {"SO": 35, "SP": 50, "SR": 8, "SLM": 0},
                    "OLM": {"SO": 21, "SP": 0, "SR": 0, "SLM": 0},
                },
                "synapses": from_yaml("simulation/config/synapses.yml"),
            },
            {
                "synapses.PYR.STIM.mechanisms.AMPA.weight": 3.0,
                "synapses.PYR.STIM.mechanisms.NMDA.weight": 3.0,
            },
        ],
    ).launch()

    network = get(
        "miv_simulator.interface.network_architecture",
        {
            "filepath": h5_types.output_filepath,
            "cell_distributions": h5_types.config.cell_distributions,
            "synapses": h5_types.config.synapses,
            "layer_extents": {
                "SO": [[0.0, 0.0, 0.0], [200.0, 200.0, 5.0]],
                "SP": [[0.0, 0.0, 5.0], [200.0, 200.0, 50.0]],
                "SR": [[0.0, 0.0, 50.0], [200.0, 200.0, 100.0]],
                "SLM": [[0.0, 0.0, 100.0], [200.0, 200.0, 150.0]],
            },
        },
    ).launch()

    network.measure_distances().launch()

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
                "axon_extents": {
                    "STIM": {"default": {"width": [200, 200], "offset": [0, 0]}},
                    "PYR": {"default": {"width": [200, 200], "offset": [0, 0]}},
                    "PVBC": {"default": {"width": [200, 200], "offset": [0, 0]}},
                    "OLM": {"default": {"width": [200, 200], "offset": [0, 0]}},
                },
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

    return graph
