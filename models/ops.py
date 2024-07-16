from models.mlp import MLP
import numpy as np
from machinable.utils import save_file
import os
import dmosopt.MOASMO as opt


def mlp(
    optimizer_cls,
    Xinit,
    Yinit,
    C,
    xlb,
    xub,
    file_path,
    options,
    # -
    mode="c+o",
    objectives=True,
    constraints=False,
    sensitivity=False,
    outlier_threshold=3.0,
    feasibility_solving=False,
    feasibility_max_iterations=50,
    feasibility_use_joint_loss=True,
    feasibility_max_steps_filter=True,
):
    x = Xinit.copy()
    y = Yinit.copy()
    yC = None
    if C is not None:
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
            num_constraints=C.shape[1] if C is not None else 0,
            num_objectives=Yinit.shape[1],
            mode=mode,
            outlier_threshold=outlier_threshold,
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
        model if objectives else None,
        model if constraints else None,
        model if sensitivity else None,
    )


def dynamic_sampling(
    file_path,
    iteration,
    evaluated_samples,
    next_samples,
    sampler,
    # ---
    samples_per_iteration=25,
    max_samples=500,
    stop_condition="convergence_condition",
    convergence_condition="iteration > 3 and max(recent('ecov', 3)) < 0.1)",
    optimizer_sampling=None,
    feasibility_solving=False,
    feasibility_max_iterations=50,
    feasibility_use_joint_loss=True,
    feasibility_max_steps_filter=True,
    # ---
    _history=[],
):
    if len(evaluated_samples) >= max_samples:
        # time-out
        return

    if len(evaluated_samples) == 0:
        # if resuming run that does not have any samples yet,
        #  request next_samples to kick things off
        return next_samples

    # train model
    x_completed = np.vstack([x.parameters for x in evaluated_samples])
    y_completed = np.vstack([x.objectives for x in evaluated_samples])
    c_completed = np.vstack([x.constraints for x in evaluated_samples])

    c_completed = (c_completed > 0).astype(int)
    feasible = c_completed.all(axis=1)

    model = MLP(
        num_parameters=x_completed.shape[1],
        num_constraints=c_completed.shape[1],
        num_objectives=y_completed.shape[1],
    )

    autoepochs = model.autoepoch(x_completed, y_completed, c_completed, verbose=1)

    model.autofit(
        x_completed,
        y_completed,
        c_completed,
        verbose=1,
        epochs=np.mean(autoepochs),
    )

    # gather stats
    scores = model.autoeval(x_completed, y_completed, c_completed)

    scores["global_feasible_ratio"] = np.mean(feasible)
    scores["feasible_ratio"] = np.mean(feasible[-samples_per_iteration:])
    scores["ecov"] = np.std(autoepochs) / np.mean(autoepochs)
    scores["autoepochs"] = autoepochs
    scores["num_samples"] = x_completed.shape[0]
    scores["iteration"] = iteration

    _history.append(scores)

    def h(key, horizon):
        return [s[key] for s in _history[-horizon:]]

    utilities = {"recent": h, "history": _history}

    scores["convergence_condition"] = is_converged = False
    if len(_history) > 0 and _history[-1]["convergence_condition"] is True:
        scores["convergence_condition"] = is_converged = True
    elif isinstance(convergence_condition, str) and eval(
        convergence_condition, scores.copy(), utilities.copy()
    ):
        scores["convergence_condition"] = is_converged = True

    # save meta-data
    save_file([os.path.dirname(file_path), "dynamic_sampling.jsonl"], scores, mode="a")

    # continue sampling?
    if isinstance(stop_condition, str) and eval(
        stop_condition, scores.copy(), utilities.copy()
    ):
        return

    # generate next samples
    candidate_samples = []

    if optimizer_sampling is not None:
        # include optimizer suggested samples
        from dmosopt.NSGA2 import NSGA2
        from dmosopt.model import Model

        optimizer = NSGA2(
            popsize=_history[0]["num_samples"] // x_completed.shape[1],
            nInput=x_completed.shape[1],
            nOutput=y_completed.shape[1],
            model=Model(),
            distance_metric=None,
        )
        local_random = np.random.default_rng()
        bounds = np.column_stack((sampler["xlb"], sampler["xub"]))
        optimizer.initialize_strategy(x_completed, y_completed, bounds, local_random)
        x_gen, _ = optimizer.generate()

        # overwrite percentage of random samples with optimizer generated samples
        k = round(samples_per_iteration * float(optimizer_sampling))
        for i, x_opt in enumerate(x_gen):
            if i > k:
                break
            candidate_samples.append(x_opt)

    if (draw := samples_per_iteration - len(candidate_samples)) > 0:
        new_samples = opt.xinit(
            samples_per_iteration,
            sampler["param_names"],
            sampler["xlb"],
            sampler["xub"],
            nPrevious=None,
            maxiter=sampler["maxiter"],
            method=sampler["method"],
        )
        for i in np.random.choice(
            np.arange(len(new_samples)), size=draw, replace=False
        ):
            candidate_samples.append(new_samples[i])
    assert len(candidate_samples) == samples_per_iteration, len(candidate_samples)
    candidate_samples = np.array(candidate_samples)

    if not is_converged or feasibility_solving is False:
        return candidate_samples

    # feasibiliy solving
    x_transformed, _ = model.make_feasible(
        candidate_samples,
        learning_rate=0.1,
        transform=[(l, u) for l, u in zip(sampler["xlb"], sampler["xub"])],
        max_iterations=feasibility_max_iterations,
        max_steps_filter=feasibility_max_steps_filter,
        use_joint_loss=feasibility_use_joint_loss,
    )

    return x_transformed
