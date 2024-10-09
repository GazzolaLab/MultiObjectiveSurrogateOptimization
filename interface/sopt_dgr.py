
from interface.dmosopt import Dmosopt
from typing import Dict
from pydantic import Field


class Dgr(Dmosopt):
    class Config(Dmosopt.Config):
        dopt_params: Dict = Field(
            default_factory=lambda: {
                "opt_id": "default",
                "obj_fun_name": "benchmarks.dgr.obj_fun",
                "problem_parameters": {
                    "fbi": 1.65,
                    "PP_weight": 1.0,
                },
                "space": {
                    "wmg": (0.1, 2.0),  # MC to GC
                    "wbg": (3.0, 5.0),  # BC to GC
                    "whg": (1.0, 5.0),  # HC to GC
                    "wbb": (0.1, 1.0),  # BC to BC
                    "wgb": (2.0, 5.0),  # GC to BC
                    "wmb": (1.0, 5.0),  # MC to BC
                    "whb": (0.1, 1.0),  # HC to BC
                    "wmm": (0.1, 1.0),  # MC to MC
                    "wgm": (0.1, 1.0),  # GC to MC
                    "wbm": (1.0, 5.0),  # BC to MC
                    "whm": (1.0, 5.0),  # HC to MC
                    "wmh": (1.0, 2.0),  # MC to HC
                    "wgh": (1.0, 2.0),  # GC to HC
                },
                "objective_names": "benchmarks.dgr.objective_names",
                # "constraint_names": "benchmarks.dgr.constraint_names",
                "feature_dtypes": "benchmarks.dgr.feature_dtypes",
                "optimizer_name": "nsga2",
                "optimizer_kwargs": [
                    {
                        "crossover_prob": 0.9,
                        "mutation_prob": 0.1,
                    },
                    {},
                ],
                "initial_method": "slh",
                "n_initial": 3,
                "initial_maxiter": 10,
                "n_epochs": 5,
                "population_size": 400,
                "num_generations": 400,
                "resample_fraction": 1.0,
                "surrogate_method_name": None,
                "surrogate_method_kwargs": {},
                "save": True,
                "save_surrogate_evals": False,
            }
        )

    def get_model(self, x=None, PP_freq="theta", **params):
        from benchmarks.dgr import DGRate

        pp = {}
        if x is None:
            pp = {**self.config.dopt_params.problem_parameters, **params}
        else:
            pp = {**self.parameter_vector_to_dict(x), **params}
        return DGRate(PP_freq=PP_freq, **pp)

    def plot_rates(self, x=None, PP_freq="theta", **params):
        import matplotlib.pyplot as plt

        network_model = self.get_model(x, PP_freq, **params)

        output = network_model.run()

        params = network_model.pars

        g, b, m, h = (output[k] for k in ["g", "b", "m", "h"])

        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(4, 5))

        ax1.plot(params["range_t"], h, color="0.5", label="HIPP")
        ax1.set_ylabel("HIPP")

        ax2.plot(params["range_t"], b, color="0.5", label="BC")
        ax2.set_ylabel("BC")

        ax3.plot(params["range_t"], m, color="0.5", label="MC")
        ax3.set_ylabel("MC")

        ax4.plot(params["range_t"], g, color="0.5", label="GC")
        ax4.set_ylabel("GC")

        ax5.plot(params["range_t"], params["PP"], color="0.5", label="PP")
        ax5.set_ylabel("PP")
        ax5.set_xlabel("Time (ms)")

        fig.tight_layout()
        fig.align_ylabels()

        return fig
