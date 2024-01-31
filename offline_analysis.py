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

for i in range(X_test.shape[0] - 1, 0, -100):
    fn = f"results/{i}.p"

    if os.path.isfile(fn):
        continue

    x_prime = X_test[i : i + 1, :]
    y_prime = y_test[i : i + 1, :]

    y_pred = model.predict(x_prime, verbose=0)

    evals = []
    for transform in ["square", "exp", "piece_exp"]:
        x_feasible, steps = model.make_feasible(
            x_prime, learning_rate=0.1, transform=transform, verbose=0
        )
        y_feasible = model.predict(x_feasible, verbose=0)
        xy_true = c.evaluate_objective_at(x_feasible[0])
        evals.append(
            {
                "transform": transform,
                "steps": steps,
                "x_feasible": x_feasible,
                "y_feasible": y_feasible,
                "xy_true": xy_true,
            }
        )

    d = {
        "i": i,
        "x_prime": x_prime,
        "y_prime": y_prime,
        "y_pred": y_pred,
        "evals": evals,
    }

    with open(fn, "wb") as f:
        pickle.dump(d, f)


# %%

# analysis
if False:
    dd = []
    for i in range(25000):
        fn = f"results/{i}.p"
        if not os.path.isfile(fn):
            continue
        with open(fn, "rb") as f:
            dd.append(pickle.load(f))

    for transform in ["square", "exp", "piece_exp"]:
        destroyed = 0
        kept_feasible = 0
        made = 0
        for d in dd:
            i = d["i"]
            ev = [e for e in d["evals"] if e["transform"] == transform][0]

            was_feasible = (d["y_prime"] > 0).all()
            believed_feasible = (d["y_pred"] > 0).all()
            made_feasible = (ev["xy_true"][-1] > 0).all()

            if not was_feasible and made_feasible:
                made += 1
            elif was_feasible and not believed_feasible and not made_feasible:
                destroyed += 1
                print(
                    "destroyed ", i, " which was believed feasible: ", believed_feasible
                )
            elif was_feasible and not believed_feasible and made_feasible:
                kept_feasible += 1

        print("TRANSFORM: ", transform, " - out of ", len(dd), " ---------")
        print("Destroyed", destroyed)
        print("Kept feasible", kept_feasible)
        print("Made feasible", made)

# %%
