from models.mlp import MLP
import numpy as np
from machinable.utils import save_file
import os

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
    feasibility_solving=False,
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
            return self.predict_objectives(x)

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

    model.autofit(x, y, yC, verbose=0, epochs=model.autoepoch(x, y, yC, verbose=0))

    scores = model.autoeval(x, y, yC)

    if isinstance(feasibility_solving, str):
        # activate on a certain condition
        feasibility_solving = bool(eval(feasibility_solving, scores.copy()))

    scores["feasibility_solving"] = feasibility_solving

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

    model.stats = {f"model_{k}": v for k, v in scores.items()}

    return (
        Optimizer.wrapped,
        model if "objective" in scope else None,
        model if "feasiblity" in scope else None,
        model if "sensitivity" in scope else None,
    )


def dynamic_sampling(
    file_path,
    iteration,
    evaluated_samples,
    next_samples,
    sampler,
    max_iterations=10,
    stop_condition="f1>0.4",
    feasibility_solving=False,
    feasibility_max_iterations=50,
    feasibility_use_joint_loss=True,
    feasibility_max_steps_filter=True,
):
    if iteration >= max_iterations:
        return

    if len(evaluated_samples) == 0:
        # if resuming run that does not have any samples yet,
        #  request next_samples to kick things off
        return next_samples

    # train model
    x_completed = np.vstack([x.parameters for x in evaluated_samples])
    y_completed = np.vstack([x.objectives for x in evaluated_samples])
    c_completed = np.vstack([x.constraints for x in evaluated_samples])

    model = MLP(
        num_parameters=x_completed.shape[1],
        num_constraints=c_completed.shape[1],
        num_objectives=y_completed.shape[1],
        joint=True,
    )

    model.autofit(
        x_completed,
        y_completed,
        c_completed,
        verbose=1,
        epochs=model.autoepoch(x_completed, y_completed, c_completed, verbose=1),
    )

    # continue sampling?

    scores = model.autoeval(x_completed, y_completed, c_completed)
    
    scores['iteration'] = iteration

    if eval(stop_condition, scores.copy()):
        return

    # generate next samples
    if isinstance(feasibility_solving, str):
        feasibility_solving = scores['feasibility_solving'] = bool(eval(feasibility_solving, scores.copy()))
        
    save_file([os.path.dirname(file_path), "dynamic_sampling.jsonl"], scores, mode="a")

    if not feasibility_solving:
        return next_samples

    x_transformed, _ = model.make_feasible(
        next_samples,
        learning_rate=0.1,
        transform=[(l, u) for l, u in zip(sampler["xlb"], sampler["xub"])],
        max_iterations=feasibility_max_iterations,
        max_steps_filter=feasibility_max_steps_filter,
        use_joint_loss=feasibility_use_joint_loss,
    )

    return x_transformed
