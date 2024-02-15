import numpy as np
import seaborn as sns
from matplotlib.colors import LogNorm
from matplotlib import pyplot as plt
from sklearn.utils import resample
from machinable import get, Index, Interface


def balance_data(X, y, sampling_strategy="auto"):
    dataset = np.hstack((X, y))
    unique_rows, counts = np.unique(y, axis=0, return_counts=True)
    if sampling_strategy == "auto":
        target_count = np.median(counts).astype(int)
    else:
        target_count = counts.max() if sampling_strategy == "over" else counts.min()

    resampled_dataset = []
    for row in unique_rows:
        class_subset = dataset[np.all(dataset[:, -y.shape[1] :] == row, axis=1)]
        if (sampling_strategy == "over" and len(class_subset) < target_count) or (
            sampling_strategy == "under" and len(class_subset) > target_count
        ):
            resampled_class_subset = resample(
                class_subset,
                replace=sampling_strategy == "over",
                n_samples=target_count,
                random_state=0,
            )
        else:
            resampled_class_subset = class_subset

        resampled_dataset.append(resampled_class_subset)

    balanced_dataset = np.vstack(resampled_dataset)

    np.random.shuffle(balanced_dataset)

    return balanced_dataset[:, : -y.shape[1]], balanced_dataset[:, -y.shape[1] :]


def split_data(X, y, train, test, balance="auto"):
    X_train = X[train].copy()
    y_train = y[train].copy()
    X_test = X[test].copy()
    y_test = y[test].copy()

    if balance:
        X_train, y_train = balance_data(X_train, y_train, sampling_strategy=balance)

    return X_train, y_train, X_test, y_test


def convert_to_k(number):
    return str(round(number / 1000)) + "k"


def constraint_map(constraints, resolution=10):
    q = (constraints.to_numpy() > 0).astype(int)
    # append global
    q = np.concatenate((q, q.all(axis=1).astype(int).reshape(-1, 1)), axis=1)

    r = np.add.reduceat(q, np.arange(0, len(q), int(len(q) / resolution)))
    r = r / r.sum(axis=1)[:, None] * 100
    sns.heatmap(
        r.T,
        annot=True,
        fmt=".1f",
        xticklabels=[
            convert_to_k((i + 1) / resolution * len(q)) for i in range(0, resolution)
        ],
        yticklabels=list(constraints.columns) + ["Global"],
        vmin=0,
        vmax=100,
        norm=LogNorm(),
        cbar_kws={"format": "%.2g"},
    )
    plt.xlabel("Problem evaluations")
    plt.title("Percentage of feasible solutions")


def globus_upload(interface, including_related: bool = True) -> None:
    storage = get("interface.storage.globus")
    storage.update(interface)

    if not including_related:
        return

    for r in interface.related().all():
        storage.update(r)


def globus_download(uuid: str) -> bool:
    index = Index.get()

    if index.find_by_id(uuid):
        return True

    storage = get("interface.storage.globus")
    target = index.local_directory(uuid)
    if not storage.retrieve(uuid, target):
        raise RuntimeError("Could not retrieve from storage")

    interface = Interface.from_directory(target)
    index.commit(interface.__model__)

    return True
