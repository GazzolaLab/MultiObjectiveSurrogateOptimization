import os

from matplotlib import pyplot as plt
from machinable import get
from machinable.utils import save_file, load_file
import pandas as pd
import numpy as np
import scienceplots


plt.style.use(["science", "nature"])
plt.rcParams.update({"figure.dpi": "300"})

np.set_printoptions(linewidth=np.inf, suppress=True)


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


def Figure():
    import pylustrator
    from matplotlib import _pylab_helpers

    pylustrator.start()

    # allow cross-cell figure collection
    _pylab_helpers.Gcf.destroy_all = lambda *args: ...


def fronts_nadir(components, **kwargs):
    return (
        pd.concat(components.map(lambda e: e.get_best(**kwargs)["y"])).max().to_list()
    )


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
            with get("interface.execution.download"):
                interface.launch()


def stash(key: str, data):
    if hasattr(data, "savefig"):
        data.savefig(f"./figures/{key}.svg", transparent=True)
        data.savefig(f"./figures/{key}.png", transparent=True)
        data.savefig(f"./figures/{key}.pdf", transparent=True)
        return

    if "." not in key:
        key += ".json"
    save_file(f"./figures/data/{key}", data)


def restore(key: str, default=None):
    if "." not in key:
        key += ".json"
    return load_file(f"./figures/data/{key}", default)
