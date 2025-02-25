from machinable import Interface, get


class ZDT(Interface):
    def launch(self):
        zdt1 = get(
            "interface.dmosopt",
            {
                "dopt_params": {
                    "opt_id": "zdt",
                    "space": {"x%d" % (i + 1): [0.0, 1.0] for i in range(30)},
                    "objective_names": ["y1", "y2"],
                    "problem_parameters": {},
                    "initial_maxiter": 10,
                    "optimizer_name": "age",
                    "n_initial": 3,
                    "population_size": 10,
                    "num_generations": 10,
                    "n_epochs": 2,
                    "save": True,
                    "obj_fun_name": "benchmarks.zdt.obj_fun",
                }
            },
        ).launch()
