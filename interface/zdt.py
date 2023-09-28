from machinable import Component

import sys, logging
import numpy as np
from dmosopt import dmosopt
from dataclasses import dataclass
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import h5py
from dmosopt.dmosopt import init_from_h5
from dmosopt.MOASMO import get_best
from dmosopt.hv import HyperVolume
from dmosopt import indicators


def zdt1(x):
    """This is the Zitzler-Deb-Thiele Function - type A
    Bound: XUB = [1,1,...]; XLB = [0,0,...]
    dim = 30
    """
    num_variables = len(x)
    f = np.zeros(2)
    f[0] = x[0]
    g = 1.0 + 9.0 / float(num_variables - 1) * np.sum(x[1:])
    h = 1.0 - np.sqrt(f[0] / g)
    f[1] = g * h
    return f


def obj_fun(pp):
    """Objective function to be minimized."""
    param_values = np.asarray([pp[k] for k in sorted(pp)])
    res = zdt1(param_values)
    logger.info(f"Iter: \t pp:{pp}, result:{res}")
    return res


def zdt1_pareto(n_points=100):
    f = np.zeros([n_points, 2])
    f[:, 0] = np.linspace(0, 1, n_points)
    f[:, 1] = 1.0 - np.sqrt(f[:, 0])
    return f


class Zdt(Component):
    @dataclass
    class Config:
        population_size: int = 200
        num_generations: int = 100
        n_epochs: int = 4

    def __call__(self):
        space = {}
        for i in range(30):
            space["x%d" % (i + 1)] = [0.0, 1.0]
        problem_parameters = {}
        objective_names = ["y1", "y2"]

        # Create an optimizer
        dmosopt_params = {
            "opt_id": self.opt_id,
            "obj_fun_name": "obj_fun",
            "obj_fun_module": self.module,
            "problem_parameters": problem_parameters,
            "space": space,
            "objective_names": objective_names,
            "population_size": self.config.population_size,
            "num_generations": self.config.num_generations,
            "initial_maxiter": 10,
            "optimizer": "age",
            "termination_conditions": True,
            "n_initial": 3,
            "n_epochs": self.config.n_epochs,
            "save_surrogate_eval": True,
            "save": True,
            "file_path": self.output_filepath,
        }

        dmosopt.run(dmosopt_params, verbose=True)

    @property
    def opt_id(self):
        return "dmosopt"

    @property
    def output_filepath(self):
        return self.local_directory("dmosopt.h5")

    def results(self):
        (
            _,
            max_epoch,
            old_evals,
            param_names,
            is_int,
            lo_bounds,
            hi_bounds,
            objective_names,
            feature_names,
            constraint_names,
            problem_parameters,
            problem_ids,
        ) = init_from_h5(self.output_filepath, None, self.opt_id, None)

        problem_id = 0

        with h5py.File(self.output_filepath, "r") as f:
            # metadata = f[f'/{self.opt_id}/metadata'][:]
            predictions = f[f"{self.opt_id}/{problem_id}/predictions"][:]
            objectives = f[f"{self.opt_id}/{problem_id}/objectives"][:]
            epochs = f[f"/{self.opt_id}/{problem_id}/epochs"][:]

        old_eval_epochs = [e.epoch for e in old_evals[problem_id]]
        old_eval_xs = [e.parameters for e in old_evals[problem_id]]
        old_eval_ys = [e.objectives for e in old_evals[problem_id]]
        x = np.vstack(old_eval_xs)
        y = np.vstack(old_eval_ys)
        old_eval_fs = None
        f = None
        if feature_names is not None:
            old_eval_fs = [e.features for e in old_evals[problem_id]]
            f = np.concatenate(old_eval_fs, axis=None)

        old_eval_cs = None
        c = None
        if constraint_names is not None:
            old_eval_cs = [e.constraints for e in old_evals[problem_id]]
            c = np.vstack(old_eval_cs)

        x = np.vstack(old_eval_xs)
        y = np.vstack(old_eval_ys)

        if len(old_eval_epochs) > 0 and old_eval_epochs[0] is not None:
            epochs = np.concatenate(old_eval_epochs, axis=None)

        n_dim = len(lo_bounds)
        n_objectives = len(objective_names)

        predictions_array = np.column_stack(
            tuple(predictions[x] for x in predictions.dtype.names)
        )
        objectives_array = np.column_stack(
            tuple(objectives[x] for x in objectives.dtype.names)
        )

        best_x, best_y, best_f, best_c, best_epoch, _ = get_best(
            x,
            y,
            f,
            c,
            len(param_names),
            len(objective_names),
            epochs=epochs,
            feasible=True,
        )

        return locals()

    def pareto_front(self):
        results = self.results()
        return np.stack((results["best_x"][:, 0], results["best_y"][:, 1])).T

    def hypervolume(self, reference):
        pf = self.pareto_front()
        indicator = indicators.Hypervolume(ref_point=reference, pf=pf)
        return indicator.do(pf)

    def igd(self, reference):
        pf = self.pareto_front()
        indicator = indicators.IGD(pf=reference)
        return indicator.do(pf)

    def pareto_plot(self):
        results = self.results()
        y, best_x, best_y = results["y"], results["best_x"], results["best_y"]

        plt.plot(y[:, 0], y[:, 1], "b.", label="evaluated points")
        plt.plot(best_x[:, 0], best_y[:, 1], "r.", label="best points")

        y_true = zdt1_pareto()
        plt.plot(y_true[:, 0], y_true[:, 1], "k-", label="True Pareto")
        plt.legend()
