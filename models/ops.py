from models.resnet import Resnet
import numpy as np
from machinable.utils import save_file
import os
import dmosopt.MOASMO as opt
import tensorflow as tf
from pprint import pprint


def joint(
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
    feasibility_solving=False,
    feasibility_max_iterations=50,
    feasibility_use_joint_loss=True,
    feasibility_max_steps_filter=True,
    save_weights=True,
    iterations=[],
):
    """
    Joint model surrogate

    # Arguments

    - mode="c+o"
        What information to use when training the model (c=constraints, o=objectives, c+o=both)
    - objectives=True
    - constraints=False
    - sensitivity=False
        Things to predict with this model
        Note that if a scope is disabled, it will fall back on the usual
        dmosopt options. For example, if you specify `"surrogate_method_name": 'gpr'`
        and set `constraints=True`, the MLP model will be used for the constraints
        but to predict the objective the usual `gpr` surrogate will be used.
    - feasibility_solving=False
        If True, the gradient information of the model will be used to push samples
        towards feasibility. This can be activated conditionally using a string,
        e.g. `'f1>0.4'` to only solve if the models F1 score is greater than 0.4
    - feasibility_max_iterations=50
        Only applies if feasibility_solving is True; number of iterations
    - feasibility_use_joint_loss=True
        Only applies if feasibility_solving is True; whether to use joint loss
    - feasibility_max_steps_filter=True
        Only applies if feasibility_solving is True; optional early stopping
    - save_weights=True
        Whether to save a checkpoint of the trained model in each epoch
    """

    tf.keras.backend.clear_session()

    x = Xinit.copy()
    y = Yinit.copy()
    yC = None
    if C is not None:
        yC = (C > 0).astype(int)

    class _Model:
        def __init__(self, model) -> None:
            self._wrapped = model

        def rank(self, x):
            # return dummy; will be reset in Optimizer
            return None

        def evaluate(self, x):
            return self.predict_objectives(x)

        def di_dict(self):
            def _reduction(s):
                a = tf.abs(s)
                n = a / tf.reduce_max(a, axis=0)
                return tf.reduce_mean(n, axis=0)

            sens = self.sensitivity(x, _reduction)
            if isinstance(sens, dict):
                # disregard constraint gradients
                sens = sens["objectives"]

            # higher sensitivity (larger gradient) results in larger di values, leading to smaller perturbations
            # lower sensitivity (smaller gradient) results in smaller di values, leading to larger perturbations
            di_crossover = 5 + (sens * 25)
            di_mutation = 5 + (sens * 45)

            if sensitivity == "cross_check":
                # invert values to cross-check effect of sensitivity
                di_crossover = 30 - (sens * 25)
                di_mutation = 50 - (sens * 45)

            return {
                "di_mutation": di_mutation,
                "di_crossover": di_crossover,
            }

        def __call__(self, *args, **kwargs):
            return self._wrapped(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    model = _Model(
        Resnet(
            num_parameters=Xinit.shape[1],
            num_constraints=C.shape[1] if C is not None else 0,
            num_objectives=Yinit.shape[1],
            mode=mode,
            xlb=xlb,
            xub=xub,
        )
    )

    model.autofit(
        x, y, yC, verbose=0, epochs=np.mean(model.autoepoch(x, y, yC, verbose=0))
    )

    scores = model.autoeval(x, y, yC)

    scores["num_samples"] = x.shape[0]

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

    model.stats = {
        f"model_{k}": v.tolist() if isinstance(v, np.ndarray) else v
        for k, v in scores.items()
    }

    if save_weights:
        model.save_weights(
            os.path.join(os.path.dirname(file_path), f"{len(iterations)}.weights.h5")
        )
        iterations.append(0)

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
    mode="c+o",
    optimizer_sampling=None,
    feasibility_solving=False,
    feasibility_max_iterations=50,
    feasibility_use_joint_loss=True,
    feasibility_max_steps_filter=True,
    verbose=1,
    # ---
    _history=[],
):
    """
    Surrogate-driven sampling

    # Arguments

    - samples_per_iteration=25
    - max_samples=500
    - stop_condition="convergence_condition"
    - convergence_condition="iteration > 3 and max(recent('ecov', 3)) < 0.1)"
    - mode="c+o"
        What information to use when training the model (c=constraints, o=objectives, c+o=both)
        Add `!` to force mode even if most of the constraint samples are equal
    - optimizer_sampling=None
        Whether to use the optimizer to suggest samples
    - feasibility_solving=False
        If True, the gradient information of the model will be used to push samples
        towards feasibility. This can be activated conditionally using a string,
        e.g. `'f1>0.4'` to only solve if the models F1 score is greater than 0.4
    - feasibility_max_iterations=50
        Only applies if feasibility_solving is True; number of iterations
    - feasibility_use_joint_loss=True
        Only applies if feasibility_solving is True; whether to use joint loss
    - feasibility_max_steps_filter=True
        Only applies if feasibility_solving is True; optional early stopping
    """
    if verbose > 0:
        print(f"Dynamic sampling: starting iteration {iteration} ({file_path})")

    if len(evaluated_samples) >= max_samples:
        # time-out
        if verbose > 0:
            print(f"Dynamic sampling reached maximum {max_samples}, terminating ...")
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

    constraint_unique_samples = np.unique(c_completed, axis=0).shape[0]
    constraint_equal_ratio = constraint_unique_samples / c_completed.shape[0]

    # check if all constraints are equal
    #  this may happen at the beginning and will make
    if constraint_unique_samples < 5 or constraint_equal_ratio > 0.1:
        if verbose > 0:
            print("Most or all constraint samples are equal")

        if "!" in mode:
            if mode.replace("!", "") == "c":
                # constraint-only training is meaningless, keep sampling
                if verbose > 0:
                    print("Continue with sampling ...")
                return next_samples

            # leave forced mode
        else:
            # fall back on objectives alone
            if verbose > 0:
                print(f"Using o-mode (overriding {mode}-mode)")
            mode = "o"

    model = Resnet(
        num_parameters=x_completed.shape[1],
        num_constraints=c_completed.shape[1],
        num_objectives=y_completed.shape[1],
        mode=mode.replace("!", ""),
        xlb=sampler["xlb"],
        xub=sampler["xub"],
    )

    autoepochs = model.autoepoch(x_completed, y_completed, c_completed, verbose=0)

    if min(autoepochs) < 5:
        if verbose > 0:
            print(
                "Invalid autoepoch, likely because of NaNs or convergence issues, sampling more ..."
            )
        return next_samples

    model.autofit(
        x_completed,
        y_completed,
        c_completed,
        verbose=0,
        epochs=np.mean(autoepochs),
    )

    # gather stats
    scores = model.autoeval(x_completed, y_completed, c_completed)

    scores.setdefault("accuracy", -1)
    scores.setdefault("precision", -1)
    scores.setdefault("recall", -1)
    scores.setdefault("f1", -1)

    scores["constraint_equal_ratio"] = constraint_equal_ratio
    scores["constraint_unique_samples"] = constraint_unique_samples
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

    scores["convergence_condition"] = False
    if len(_history) > 0 and _history[-1]["convergence_condition"] is True:
        scores["convergence_condition"] = True
    elif isinstance(convergence_condition, str) and eval(
        convergence_condition, scores.copy(), utilities.copy()
    ):
        scores["convergence_condition"] = True

    if isinstance(feasibility_solving, str):
        feasibility_solving = bool(eval(feasibility_solving, scores.copy()))
    scores["feasibility_solving"] = feasibility_solving

    if verbose > 0:
        print("Dynamic sampling evaluation:")
        pprint(scores)

    # save meta-data
    save_file([os.path.dirname(file_path), "dynamic_sampling.jsonl"], scores, mode="a")

    # continue sampling?
    if isinstance(stop_condition, str) and eval(
        stop_condition, scores.copy(), utilities.copy()
    ):
        if verbose > 0:
            print("Dynamic sampling stop condition reached, stopping ...")
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

    if feasibility_solving is False:
        return candidate_samples

    # feasibiliy solving
    x_transformed, _ = model.make_feasible(
        candidate_samples,
        learning_rate=0.001,
        max_iterations=feasibility_max_iterations,
        max_steps_filter=feasibility_max_steps_filter,
        use_joint_loss=feasibility_use_joint_loss,
    )

    if verbose > 0:
        print("Feasibility solving completed ...")
        print("Inputs:")
        pprint(candidate_samples[:5])
        print("=" * 20)
        print("Transformed:")
        pprint(x_transformed[:5])
        print("Delta:")
        pprint(candidate_samples[:5] - x_transformed[:5])

    return x_transformed
