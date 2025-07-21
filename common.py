import os

from matplotlib import pyplot as plt
from machinable import get
from machinable.utils import save_file, load_file
import pandas as pd
import numpy as np
import scienceplots
from scipy import stats
from sklearn import metrics

plt.style.use(["science", "nature"])
plt.rcParams.update({"figure.dpi": "300"})

np.set_printoptions(linewidth=np.inf, suppress=True)

FIGURE_EXPORT_DIRECTORY = os.environ.get("FIGURES_MSO", "./figures")


colors = {
    "blue": "#0C5DA5",
    "green": "#00B945",
    "yellow": "#FF9500",
    "red": "#FF2C00",
    "violet": "#845B97",
    "black": "#474747",
    "grey": "#9e9e9e",
}

icolors = [c for c in colors.values()]

mcolors = {
    k: icolors[i]
    for i, k in enumerate(
        [
            "gpr",
            "megp",
            "o-resnet",
            "c+o-resnet",
            "o-fttransformer",
            "c+o-fttransformer",
        ]
    )
}


def fronts_nadir(components, **kwargs):
    return (
        pd.concat(components.map(lambda e: e.get_best(**kwargs)["y"])).max().to_list()
    )


def compute_auc_hvs(hypervolumes):
    aucs = {}
    for p, models in hypervolumes.items():
        aucs.setdefault(p, {})
        for model_name, epoch_hvs in models.items():
            x = []
            ys = {}
            for epoch, hvs in epoch_hvs.items():
                x.append(int(epoch))
                for i in range(len(hvs)):
                    ys.setdefault(i, [])
                    ys[i].append(hvs[i])

            aucs[p][model_name] = [
                metrics.auc(x, y) for y in ys.values() if len(y) == len(x)
            ]

    return aucs


def population_nadirs(*interface):
    if len(interface) > 1:
        nds = tuple(population_nadirs(i) for i in interface)
        return {
            p: [float(np.max(vals)) for vals in zip(*(n[p] for n in nds))]
            for p in nds[0]
        }

    n = {}
    for p in interface[0].populations():
        experiments = interface[0].components.filter(lambda x: p in x.label())
        n[p] = fronts_nadir(experiments)
    return n


def status(interface):
    with get("interface.execution.status"):
        interface.launch()


def globus_download(interface, force=False):
    if not os.path.isfile(interface.components[0].output_filepath) or force:
        with get("interface.storage.globus"):
            with get("interface.execution.download", {"force": force}):
                interface.launch()


def stash(key: str, data):
    if hasattr(data, "savefig"):
        data.savefig(
            os.path.join(FIGURE_EXPORT_DIRECTORY, f"{key}.svg"), transparent=True
        )
        data.savefig(
            os.path.join(FIGURE_EXPORT_DIRECTORY, f"{key}.png"), transparent=True
        )
        data.savefig(
            os.path.join(FIGURE_EXPORT_DIRECTORY, f"{key}.pdf"), transparent=True
        )
        return

    if "." not in key:
        key += ".json"
    save_file(f"./figures/data/{key}", data)


def restore(key: str, default=None):
    if "." not in key:
        key += ".json"
    return load_file(f"./figures/data/{key}", default)
