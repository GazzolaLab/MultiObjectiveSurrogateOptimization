from machinable import Interface, get
from miv_simulator import mechanisms


class Microcircuit(Interface):
    def launch(self):
        repo = "benchmarks/motoneuron_culture"
        network = get(
            "miv_simulator.interface.network",
            {
                "config_filepath": f"{repo}/config/Microcircuit.yaml",
                "mechanisms_path": mechanisms.compile(f"{repo}/mechanisms"),
                "template_path": f"{repo}/templates",
                "morphology_path": f"{repo}/morphology",
            },
        )

        if not network.cached():
            network.launch()
            return self

        if get(
            "interface.sopt_network",
            [
                {
                    "dopt_params.obj_fun_init_args": {
                        **network.neural_h5.files(),
                        "cell_types": network.source_config.cell_types,
                        "synapses": network.source_config.synapses,
                        "mechanisms": network.config.mechanisms_path,
                        "templates": network.config.template_path,
                    },
                },
                {"dopt_params.obj_fun_init_args.cell_types.PYR.template": "SC_nrn"},
                f"""~from_clamp(
                    "{repo}/config/Network_Clamp.yaml",
                    target_populations=['PYR'], 
                    objective_names=['PYR firing rate', 'PYR fraction active']
                )""",
                {"controller_init_fun_args.subworld_size": 8},
            ],
        ).future():
            print("done!")
