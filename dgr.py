# %%
import os
from machinable import get
from matplotlib import pyplot as plt

get("machinable.index", os.environ.get("STORAGE", None)).__enter__()

# %%
experiment = get(
    "interface.dg_rate",
    {
        "dopt_params": {
            "n_epochs": 5,
        },
    },
)

# %%
with get("interface.execution.local", {"ranks": 8}):
    experiment.launch()

# %%
fig = experiment.plot_rates()
plt.show()
# %%
