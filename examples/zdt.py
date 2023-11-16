# %% [markdown]
# ## ZDT1

# %%
import os, pathlib
from machinable import get

# switch to repo root
if project_directory := next(
    (
        path
        for path in pathlib.Path(os.getcwd()).resolve().parents
        if (path / ".git").is_dir()
    ),
    None,
):
    get("machinable.project", str(project_directory)).__enter__()

# %%
with get("interface.execution.slurm", {"ranks": 8}):
    zdt1 = get(
        "miv_simulator.interface.optimize",
        {
            "optimizer": {
                "opt_id": "zdt",
                "space": {"x%d" % (i + 1): [0.0, 1.0] for i in range(30)},
                "objective_names": ["y1", "y2"],
                "problem_parameters": {},
                "initial_maxiter": 10,
                "optimizer": "age",
                "termination_conditions": True,
                "n_initial": 3,
                "population_size": 200,
                "num_generations": 200,
                "save_surrogate_eval": True,
                "n_epochs": 2,
                "save": True,
                "obj_fun_name": "obj_fun",
                "obj_fun_module": "_zdt1",
            }
        },
    ).launch()

zdt1.execution.stream_output()

# %%
zdt1.pareto_plot()

# %%
x = [10, 50, 100, 200, 300]
y = []
y2 = []

with get("interface.execution.slurm", {"nodes": 4, "ranks": 32, "partition": "normal"}):
    for population_size in x:
        zdt = get(
            zdt1.module,
            zdt1.version() + [{"optimizer.population_size": population_size}],
        ).launch()
        if not zdt.cached():
            continue
        hv = zdt.hypervolume([11, 11])
        igd = zdt.igd([11, 11])
        y.append(hv)
        y2.append(igd)

print(x)
print(y)
print(y2)
