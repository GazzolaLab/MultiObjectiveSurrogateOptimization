# %% [markdown]
# ## ZDT1

# %%

from machinable import get

with get("interface.execution.slurm", {"ranks": 8}):
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
                "obj_fun_name": "benchmarks.zdt1.obj_fun",
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
            zdt1.version() + [{"sopt_params.population_size": population_size}],
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
