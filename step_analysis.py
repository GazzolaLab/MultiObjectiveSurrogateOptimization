# %%
import os
import pickle
import multiprocessing
import numpy as np
from matplotlib import pyplot as plt
from machinable import get
from dmosopt.MOASMO import xinit
from models.mlp import MLP, apply_bounds
import tensorflow as tf

# %%
mn = get("interface.motoneuron")

model = MLP(
    num_parameters=len(mn.config.dopt_params.space.keys()),
    num_constraints=len(mn.config.dopt_params.constraint_names),
    num_objectives=len(mn.config.dopt_params.objective_names),
)

# %%

model.load_weights("model.h5")

# %%

name = "first"

if os.path.isfile(f"results/step_analysis/{name}.p"):
    with open(f"results/step_analysis/{name}.p", "rb") as f:
        dd = pickle.load(f)

        # as subplots
        fig, axs = plt.subplots(6, 1, figsize=(10, 15), sharex=True)
        for ei in range(5):
            e = dd["evals"][ei]
            steps = dd["steps"][ei]
            trajectory = dd["trajectory"]
            x = np.arange(len(e))
            y = [(_e[-1] > 0.99).sum() if k <= steps +1000000 else np.nan for k, _e in enumerate(e)]
            y_model = [(model.predict(_e[ei], verbose=0) > 0.99).sum() for _e in trajectory]
            axs[ei].plot(x, y, label=f"True", )
            axs[ei].plot(np.arange(len(y_model)), y_model, label=f"Prediction", color="deeppink")
            axs[ei].set_yticks(range(len(e[0][-1]) + 1))
            axs[ei].set_ylim(0, len(e[0][-1]) + 1)
            axs[ei].hlines(
                len(e[0][-1]), 0, len(e), linestyles="dashed", colors="green", label="Break-even"
            )
            if steps < len(e):
                axs[ei].vlines(
                    steps+1,
                    0,
                    len(e[0][-1]),
                    linestyles="dashed",
                    colors="orange",
                    label="Converged early",
                )
            axs[ei].set_ylabel("Number of fulfilled constraints")
        axs[-2].set_xlabel("Updates")
        axs[5].axis("off")
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.1, 0.1))
        plt.tight_layout()
        plt.show()
        

# %%
    exit()
# %%


# %%
space_bounds = list(mn.config.dopt_params.space.values())

print("Generating samples...")
sample = xinit(
    nEval=10,
    param_names=list(mn.config.dopt_params.space.keys()),
    xlb=np.array(space_bounds)[:, 0],
    xub=np.array(space_bounds)[:, 1],
    nPrevious=None,
    maxiter=0,
    method="slh",
)[:5, :]


print("Making feasible...")


def feasible_optimization(self, X, learning_rate=0.1, max_iterations=100):
    if len(X.shape) == 1:
        X = X.reshape(1, -1)

    for layer in self.layers:
        layer.trainable = False

    input_sample = tf.Variable(
        initial_value=X,
        dtype=tf.float32,
        name="inverse_X",
    )
    steps = tf.Variable(
        initial_value=tf.ones([X.shape[0]], dtype=np.int32) * -1,
        dtype=tf.int32,
        name="steps",
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.BinaryFocalCrossentropy()

    points = [X]

    iteration = 0
    while True:
        with tf.GradientTape() as tape:
            tape.watch(input_sample)
            z = input_sample
            prediction = self(z)
            loss = loss_fn(
                tf.constant(
                    np.ones([input_sample.shape[0], self.num_constraints]),
                    dtype=tf.float32,
                ),
                prediction,
            )

        if iteration > max_iterations:
            break

        grads = tape.gradient(loss, input_sample)

        is_feasible = tf.math.reduce_all(prediction > 0.99, axis=1)

        # record number of steps for feasible samples
        steps = tf.where(is_feasible, steps, iteration)

        # zero out grads for samples that are feasible
        is_feasible_where = tf.tile(
            tf.expand_dims(is_feasible, axis=1), [1, grads.shape[1]]
        )
        grads = tf.where(is_feasible_where, tf.zeros_like(grads), grads)

        optimizer.apply_gradients([(grads, input_sample)])

        input_sample.assign(apply_bounds(input_sample, space_bounds))

        points.append(input_sample.numpy())

        iteration += 1

    for layer in self.layers:
        layer.trainable = True

    return points, steps.numpy()


trajectory, steps = feasible_optimization(
    model, sample, learning_rate=0.1, max_iterations=50
)

# %%
evals = [[] for _ in range(len(steps))]
for i, t_sample in enumerate(trajectory):

    def _ev(a):
        x, done = a
        if done:
            return x
        return mn.evaluate_objective_at(x)

    cutoff = i > steps

    processes = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=processes) as pool:
        results = pool.map(
            _ev,
            [
                (x, False) if (not cutoff[k]) else (evals[k][-1], True)
                for k, x in enumerate(t_sample)
            ],
        )

    for i, r in enumerate(results):
        evals[i].append(r)

# %%

with open(f"results/step_analysis/{name}.p", "wb") as f:
    pickle.dump(
        {
            "trajectory": trajectory,
            "steps": steps,
            "evals": evals,
        },
        f,
    )
