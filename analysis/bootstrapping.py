# %%
import os
import numpy as np
from machinable import get, Component
from dmosopt.MOASMO import xinit
from models.mlp import MLP
import multiprocessing
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split


class SamplingExperiment(Component):
    class Config(BaseModel):
        subsample_size: int = 1000
        total_size: int = 25000
        epochs: int = 100
        target_f1: float = 0.6
        use_feasibility_model: float = Field(
            0.1,
            title="Threshold for using feasibility model to make samples feasible",
            description="Set to 1.1 to not use the model at all",
        )
        validation_split: float = 0.2

    def __call__(self):
        benchmark = self.uses[0]
        space_bounds = list(benchmark.config.dopt_params.space.values())

        def get_samples(n: int):
            # standard latin hypercube sampling
            return xinit(
                nEval=n,
                param_names=list(benchmark.config.dopt_params.space.keys()),
                xlb=np.array(space_bounds)[:, 0],
                xub=np.array(space_bounds)[:, 1],
                nPrevious=None,
                maxiter=10,
                method="slh",
            )[:n, :]

        def evaluate_objective(samples, processes=multiprocessing.cpu_count() - 1):
            with multiprocessing.Pool(processes=processes) as pool:
                return pool.map(benchmark.evaluate_objective_at, samples)

        def get_model():
            return MLP(
                num_parameters=len(benchmark.config.dopt_params.space.keys()),
                num_constraints=len(benchmark.config.dopt_params.constraint_names),
                num_objectives=len(benchmark.config.dopt_params.objective_names),
            )

        model = None

        x = self.load_file("x.p", default=None)
        yC = self.load_file("yC.p", default=None)
        evals = self.load_file("evals.p", default=[])
        results = self.load_file("results.json", default=[])
        for iteration in range(self.config.total_size // self.config.subsample_size):
            print("Iteration ", iteration)

            # generate random samples of the search space
            samples = get_samples(self.config.subsample_size)

            if (
                len(results) > 0
                and results[-1]["f1"] > self.config.use_feasibility_model
                and model is not None
            ):
                # make the samples feasible according to the latest model
                x, steps = model.make_feasible(
                    x,
                    learning_rate=0.1,
                    transform=space_bounds,
                    verbose=0,
                    max_iterations=50,
                )

            # evaluate the samples to get training data
            evaluated_samples = evaluate_objective(samples)
            evals.append(evaluated_samples)
            feasibility = np.array([sample[-1] > 0.99 for sample in evaluated_samples])

            if x is None:
                x = samples
                yC = feasibility
            else:
                x = np.concatenate((x, samples), axis=0)
                yC = np.concatenate((yC, feasibility), axis=0)

            # train the model
            model = get_model()

            X_train, X_test, y_train, y_test = train_test_split(x, yC, test_size=0.2)
            model.fit(
                X_train,
                y_train,
                epochs=self.config.epochs,
                batch_size=2048,
                validation_split=self.config.validation_split,
                verbose=0,
            )
            metrics = model.eval(X_test, y_test)

            metrics["iteration"] = iteration
            metrics["X_train_samples"] = X_train.shape[0]
            metrics["X_test_samples"] = X_test.shape[0]

            results.append(metrics)

            self.save_file("x.p", x)
            self.save_file("yC.p", yC)
            self.save_file("evals.p", evals)
            self.save_file("results.json", results)

            print("Metrics: ", metrics)

            if metrics["f1"] > self.config.target_f1:
                break


# %%

motoneuron_benchmark = get("interface.motoneuron", "~from_protocol")

experiment = get(
    SamplingExperiment,
    {"subsample_size": 100, "use_feasibility_model": 0.05},
    uses=motoneuron_benchmark,
)

experiment.launch()

# plot results

print(experiment.load_file("results.json"))
