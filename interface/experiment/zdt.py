from machinable import Interface, get
from matplotlib import pyplot as plt


class ZDT(Interface):
    def launch(self):
        zdt1 = [
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
        ]

        if experiment := get(zdt1).future():
            print("Pareto front:")
            print(experiment.get_best()["y"])

        x = [10, 20]
        y = []
        for population_size in x:
            if experiment := get(
                zdt1, {"dopt_params.population_size": population_size}
            ).future():
                hv = experiment.hypervolume([11, 11])
                y.append(hv)

        if self.future():
            plt.plot(x, y)
            plt.xlabel("Population")
            plt.ylabel("Hypervolume")
            plt.show()
