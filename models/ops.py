from models.mlp import MLP
import numpy as np


def mlp(
    optimizer_cls,
    Xinit,
    Yinit,
    C,
    xlb,
    xub,
    file_path,
    options,
    scope,
    joint=True,
    feasibility_solving=True,
    feasibility_max_iterations=50,
    feasibility_use_joint_loss=True,
    feasibility_max_steps_filter=True,
):
    x = Xinit.copy()
    y = Yinit.copy()
    yC = (C > 0).astype(int)

    class Model:
        def __init__(self, model) -> None:
            self._wrapped = model

        def rank(self, x):
            # return dummy; will be reset in Optimizer
            return None

        def evaluate(self, x):
            y_pred = self.predict(x)
            yR = self.norm_output(y_pred["objectives"], inverse=True)
            return np.nan_to_num(yR)

        def di_dict(self):
            # TODO: self.sensitivity()
            return {
                "di_mutation": None,
                "di_crossover": None,
            }

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    model = Model(
        MLP(
            num_parameters=Xinit.shape[1],
            num_constraints=C.shape[1],
            num_objectives=Yinit.shape[1],
            joint=joint,
        )
    )

    model.autofit(x, y, yC)

    class Optimizer:
        def __init__(self, optimizer) -> None:
            self._wrapped = optimizer

            # we do not use the ranking
            self._wrapped.x_distance_metrics = None

        def generate_initial(self, *args, **kwargs):
            x = self._wrapped.generate_initial(*args, **kwargs)

            x = self.sampling_modifier(x)

            return x

        def generate_strategy(self, *args, **kwargs):
            x_gen, state_gen = self._wrapped.generate_strategy(*args, **kwargs)

            x_gen = self.sampling_modifier(x_gen)

            return x_gen, state_gen

        def sampling_modifier(self, samples):
            if not feasibility_solving:
                return samples
            x_transformed, _ = model.make_feasible(
                samples,
                learning_rate=0.1,
                transform=[(l, u) for l, u in zip(xlb, xub)],
                max_iterations=feasibility_max_iterations,
                max_steps_filter=feasibility_max_steps_filter,
                use_joint_loss=feasibility_use_joint_loss,
            )
            return x_transformed

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

        @classmethod
        def wrapped(cls, *args, **kwargs):
            return cls(optimizer_cls(*args, **kwargs))

    return (
        Optimizer.wrapped,
        model if "objective" in scope else None,
        model if "feasiblity" in scope else None,
        model if "sensitivity" in scope else None,
    )
