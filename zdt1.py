# %% [markdown]
# ## ZDT1

# %%
import os
from machinable import get

get("machinable.index", os.environ["STORAGE"]).__enter__()

# %%

from machinable import get

with get("interface.execution.slurm", {"nodes": 4, "ranks": 32, "partition": "normal"}):
    # with get("interface.execution.local", {"mpi": "ibrun"}):
    zdt1 = get(
        "interface.dmosopt",
        {
            "dopt_params": {
                "opt_id": "zdt",
                "space": {"x%d" % (i + 1): [0.0, 1.0] for i in range(30)},
                "objective_names": ["y1", "y2"],
                "problem_parameters": {},
                "initial_maxiter": 10,
                "optimizer": "age",
                "n_initial": 3,
                "population_size": 200,
                "num_generations": 200,
                "save_surrogate_evals": True,
                "n_epochs": 2,
                "save": True,
                "obj_fun_name": "benchmarks.zdt.obj_fun",
            }
        },
    ).launch()

# %%
#!code {zdt1.execution.output_filepath()}

# %%
from matplotlib import pyplot as plt

zdt1.pareto_plot()
plt.show()

# %%
x = [50, 100, 200, 300]
y = []
y2 = []
import os

with get("interface.execution.slurm", {"nodes": 4, "ranks": 32, "partition": "normal"}):
    for population_size in x:
        zdt = get(
            zdt1.module,
            zdt1.version() + [{"dopt_params.population_size": population_size}],
        ).launch()
        if not zdt.cached():
            y.append(float("NaN"))
            y2.append(float("NaN"))
            continue
        hv = zdt.hypervolume([11, 11])
        igd = zdt.igd([11, 11])
        y.append(hv)
        y2.append(igd)

print(x)
print(y)
print(y2)

plt.plot(x, y)
plt.xlabel("Population")
plt.ylabel("Hypervolume")
plt.show()
plt.plot(x, y2)
plt.xlabel("Population")
plt.ylabel("IGD")
plt.show()

# %%

x = ["nsga2", "age", "smpso", "cmaes"]
y = []
y2 = []
import os

with get("interface.execution.slurm", {"nodes": 4, "ranks": 32, "partition": "normal"}):
    for optimizer in x:
        zdt = get(
            zdt1.module,
            zdt1.version() + [{"dopt_params.optimizer": optimizer}],
        ).launch()
        if not zdt.cached():
            y.append(float("NaN"))
            y2.append(float("NaN"))
            continue
        hv = zdt.hypervolume([11, 11])
        igd = zdt.igd([11, 11])
        y.append(hv)
        y2.append(igd)

print(x)
print(y)
print(y2)

plt.bar(x, y)
plt.title("Hypervolume")
plt.show()
plt.bar(x, y2)

# %%
