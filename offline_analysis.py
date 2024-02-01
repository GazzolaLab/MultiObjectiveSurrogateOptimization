# %%

import os
import pickle
from machinable import get, Component
from utils import split_data, constraint_map

get("machinable.index", os.environ["STORAGE"]).__enter__()

experiments = {
    "baseline": "049af43f0bf2-0008-82e7-0961-3727a560",
    "t90k": "b567456b327c-0008-1e77-c295-ec96a560",
    "t1": "004e67e5f21f-0008-2627-0535-5f16a560",
    "t2": "1dfb14537500-0008-2f97-9075-5f16a560",
    "t3": "d9f607275525-0008-48a7-2995-5f16a560",
}

# %%

c = Component.find_by_id(experiments["t1"])
data = c.load_h5()

X = data["parameters"].to_numpy()
C = data["constraints"].to_numpy()
objectives = data["objectives"].to_numpy()
yC = (C > 0).astype(int)

# %%

# constraint_map(data["constraints"][:50000])

# %%
n_initial = 25000

X_train, y_train, X_test, y_test = split_data(
    X, yC, train=slice(0, n_initial), test=slice(25000, 2 * 25000), balance=None
)

# %%
from models.mlp import MLP

model = MLP(
    num_parameters=X.shape[1],
    num_constraints=yC.shape[1],
    num_objectives=objectives.shape[1],
)

# %%
model.summary()
# model.save_weights("model.h5")

# %%
model.load_weights("model.h5")

# %%

# model.load_weights('model.h5')
# model.interactive()
# model.fit(
#     X_train, y_train, epochs=1000, batch_size=2048, validation_split=0.2, verbose=0
# )

# %%

import multiprocessing
from sklearn.utils import gen_batches

batch_size = 32
threshold = 100

for i, batch in enumerate(gen_batches(n_initial, batch_size)):
    fn = f"results/{i}.p"

    if os.path.isfile(fn):
        continue

    x_prime = X_test[batch, :]
    y_prime = y_test[batch, :]

    y_pred = model.predict(x_prime, verbose=0)

    x_feasible, steps = model.make_feasible(
        x_prime,
        learning_rate=0.1,
        transform=list(c.config.dopt_params.space.values()),
        verbose=0,
        max_iterations=threshold + 1,
    )

    y_feasible = model.predict(x_feasible, verbose=0)

    processes = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=processes) as pool:
        evals = pool.map(c.evaluate_objective_at, x_feasible)

    d = {
        "i": i,
        "x_prime": x_prime,
        "y_prime": y_prime,
        "y_pred": y_pred,
        "x_feasible": x_feasible,
        "y_feasible": y_feasible,
        "steps": steps,
        "evals": evals,
    }

    with open(fn, "wb") as f:
        pickle.dump(d, f)


# %%
analysis = False
# %%
if analysis:
    import numpy as np
    import pickle
    from sklearn.utils import gen_batches

    batch_size = 32
    threshold = 100
    total = 0
    total_was_feasible = 0
    total_made_feasible = 0
    total_destroyed = 0
    total_kept_feasible = 0

    for i, batch in enumerate(gen_batches(n_initial, batch_size)):
        fn = f"results/{i}.p"

        if not os.path.isfile(fn):
            break

        with open(fn, "rb") as f:
            d = pickle.load(f)

        was_feasible = (d["y_prime"] > 0).all(axis=1)
        believed_feasible = (d["y_pred"] > 0).all(axis=1)
        is_feasible = np.array([(e[-1] > 0).all() for e in d["evals"]])

        destroyed = np.zeros_like(was_feasible, dtype=int)
        kept_feasible = np.zeros_like(was_feasible, dtype=int)
        made_feasible = np.zeros_like(was_feasible, dtype=int)

        made_feasible[(~was_feasible) & is_feasible] = 1
        destroyed[(was_feasible) & (~believed_feasible) & (~is_feasible)] = 1
        kept_feasible[(was_feasible) & (~believed_feasible) & is_feasible] = 1

        total += len(d["y_prime"])
        total_was_feasible += was_feasible.sum()
        total_made_feasible += made_feasible.sum()
        total_destroyed += destroyed.sum()
        total_kept_feasible += kept_feasible.sum()

    print("Total", total)
    print("Was feasible", total_was_feasible)
    print("Destroyed", total_destroyed)
    print("Kept feasible", total_kept_feasible)
    print("Made feasible", total_made_feasible)

# %%
n_initial = 25000
analysis = True
# %%
