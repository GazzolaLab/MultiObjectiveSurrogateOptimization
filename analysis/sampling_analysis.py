# %%
import os
import numpy as np
from machinable import get
from dmosopt.MOASMO import xinit
from models.mlp import MLP

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
# with get("interface.execution.local"):
#     mn.new().launch()
# %%

min_bound = 0.00001
max_bound = 0.3
broad_bounds = {
    "gc": [min_bound, 2.0],
    "soma_gmax_Na": [min_bound, max_bound],
    "soma_gmax_K": [min_bound, max_bound],
    "soma_gmax_KCa": [min_bound, max_bound / 5],
    "soma_gmax_CaN": [min_bound, max_bound / 5],
    "soma_g_pas": [min_bound, max_bound / 5],
    "dend_gmax_CaL": [min_bound, max_bound / 5],
    "dend_gmax_CaN": [min_bound, max_bound / 5],
    "dend_gmax_KCa": [min_bound, max_bound / 5],
    "dend_g_pas": [min_bound, max_bound / 5],
    "cm_ratio": [1.0, 40.0],
}


space_bounds = list(mn.config.dopt_params.space.values())
# space_bounds = list(broad_bounds.values())


# %%
import json
import pickle
from collections import defaultdict
import multiprocessing
from sklearn.utils import gen_batches

batch_size = 23  # 2**8

for i in range(1):
    print("Generating samples...")
    sample = xinit(
        nEval=batch_size,
        param_names=list(mn.config.dopt_params.space.keys()),
        xlb=np.array(space_bounds)[:, 0],
        xub=np.array(space_bounds)[:, 1],
        nPrevious=None,
        maxiter=0,
        method="slh",
    )

    print("Making feasible...")
    x_feasible, steps = model.make_feasible(
        sample, learning_rate=0.1, transform=space_bounds, verbose=0, max_iterations=100
    )

    processes = multiprocessing.cpu_count() // 2
    with multiprocessing.Pool(processes=processes) as pool:
        results = pool.map(mn.evaluate_objective_at, [x for x in x_feasible])

    d = {
        "batch_size": batch_size,
        "sample": sample,
        "x_feasible": x_feasible,
        "steps": steps,
        "results": results,
    }

    with open(f"results/sampling_analysis/{23*11}.p", "wb") as f:
        pickle.dump(d, f)


# %%
import pickle

with open(f"results/sampling_analysis/{23*11}.p", "rb") as f:
    d = pickle.load(f)
# %%

total = 0
feasible = 0
fs = []
ss = []
for i, (r, s) in enumerate(zip(d["results"], d["steps"])):
    is_feasible = (r[-1] > 0.99).all()
    print(f"{i}: {is_feasible} - {s}")
    if is_feasible:
        feasible += 1
    if s < 100:
        fs.append(int(is_feasible))
        ss.append(s)
    total += 1
    if i >= 250 - 1:
        break

print(f"Feasible: {feasible}/{total} ({feasible/total*100:.2f}%)")
# %%

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.DataFrame({"class": fs, "feature": ss})

plt.rcParams.update(
    {"font.size": 15, "text.usetex": True, "font.family": "STIXGeneral"}
)

fig, ax = plt.subplots()
ax.violinplot(
    dataset=[
        df[df["class"] == 0]["feature"].values,
        df[df["class"] == 1]["feature"].values,
    ],
    showmeans=False,
    showmedians=True,
)

ax.set_xticks([1, 2])
ax.set_xticklabels(["Infeasible", "Feasible"])

ax.set_ylabel("Steps to convergence")
# ax.set_title('Violin plot')

plt.show()
# %%
fig, ax = plt.subplots()
ax.boxplot(
    [df[df["class"] == 0]["feature"].values, df[df["class"] == 1]["feature"].values],
    labels=["Infeasible", "Feasible"],
)
ax.set_ylabel("Steps to convergence")
# ax.set_title('Box plot')

plt.show()
# %%
