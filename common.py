import os

from machinable import get
from machinable.utils import save_file, load_file
import pandas as pd


def fronts_nadir(components, **kwargs):
    return (
        pd.concat(components.map(lambda e: e.get_best(**kwargs)["y"])).max().to_list()
    )


def status(interface):
    with get("interface.execution.status"):
        interface.launch()


def fetch(interface):
    if not os.path.isfile(interface.components[0].output_filepath):
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
