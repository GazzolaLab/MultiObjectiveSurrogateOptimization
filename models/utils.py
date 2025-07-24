import numpy as np
from matplotlib.colors import LogNorm
from matplotlib import pyplot as plt
from sklearn.utils import resample
import os
from scipy import stats
from machinable.utils import save_file, load_file
import time


class SwapQueue:
    def __init__(self, directory):
        self.directory = directory

    def put(self, key, data):
        save_file([self.directory, f"{key}.p"], data)

    def get(self, key):
        while True:
            payload = load_file([self.directory, f"{key}.p"], None)
            if payload is not None:
                return payload
            time.sleep(5)

    def send(self, data):
        self.put("send", data)
        payload = self.get("receive")
        self.done("receive")
        return payload

    def receive(self, data=None):
        if data is None:
            return self.get("send")

        self.put("receive", data)
        self.done("send")

    def done(self, key):
        file_path = os.path.join(self.directory, f"{key}.p")
        if os.path.exists(file_path):
            os.remove(file_path)


def preprocess(x, y, yC=None, remove_outliers=False, nan="remove"):
    if nan == "max":
        # replace NaNs with maximum
        m = np.max(np.nan_to_num(y), axis=0)
        for c in range(y.shape[1]):
            y[:, c] = np.nan_to_num(y[:, c], nan=2 * m[c])
    elif nan == "remove":
        r = ~np.any(np.isnan(y), axis=1)
        x = x[r]
        y = y[r]
        if yC is not None:
            yC = yC[r]
    else:
        raise ValueError("Invalid nan mode")

    # filter outliers
    if remove_outliers is True:
        remove_outliers = 2
    mask = slice(None)
    if remove_outliers is not False:
        ylog = np.log(y + 1)
        ylmean = np.mean(ylog, axis=0)
        ylstd = np.std(ylog, axis=0)
        zscores = (ylog - ylmean) / ylstd
        # filter values above the threshold (high outliers)
        outlier = np.any(zscores > float(remove_outliers), axis=1)
        mask = ~outlier

    if yC is None:
        return x[mask], y[mask], yC

    return x[mask], y[mask], yC[mask]


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
    import seaborn as sns

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


def epsilon_indicator(pf_a, pf_b, indicator_type="additive"):
    """
    Calculates the epsilon-indicator I(A, B).

    This metric measures the minimum factor (multiplicative) or term (additive)
    by which every solution in Pareto Front B (pf_b) must be worsened in
    every objective to be weakly dominated by at least one solution in
    Pareto Front A (pf_a).

    This implementation assumes all objectives are to be MINIMIZED.

    To assess if A and B are equivalent, you can compute the indicator in both directions:
        1. `eps_A_B = epsilon_indicator(pf_a, pf_b)`
        2. `eps_B_A = epsilon_indicator(pf_b, pf_a)`
    If both values are very close to 0 (additive) or 1 (multiplicative),
    the fronts can be considered highly equivalent.

    Args:
        pf_a (np.ndarray): The first Pareto front (shape: [n_solutions_A, n_objectives]).
                           This is the reference front.
        pf_b (np.ndarray): The second Pareto front (shape: [n_solutions_B, n_objectives]).
                           This is the front being evaluated against pf_a.
        indicator_type (str): The type of indicator, either 'additive' or 'multiplicative'.
                              Defaults to 'additive'.

    Returns:
        float: The epsilon-indicator value.
               - For I(A, B), a smaller value is better.
               - A value <= 0 (additive) or <= 1 (multiplicative) means A dominates B.
    """
    if not isinstance(pf_a, np.ndarray) or not isinstance(pf_b, np.ndarray):
        raise TypeError("Inputs must be NumPy arrays.")
    if pf_a.ndim != 2 or pf_b.ndim != 2:
        raise ValueError("Input arrays must be 2-dimensional.")
    if pf_a.shape[1] != pf_b.shape[1]:
        raise ValueError("Pareto fronts must have the same number of objectives.")
    if pf_a.shape[0] == 0 or pf_b.shape[0] == 0:
        return np.inf

    # Reshape for broadcasting: pf_a -> (n_a, 1, m), pf_b -> (1, n_b, m)
    # This allows comparing every point in A with every point in B efficiently.
    pf_a_reshaped = pf_a[:, np.newaxis, :]
    pf_b_reshaped = pf_b[np.newaxis, :, :]

    # Calculate element-wise differences or ratios
    if indicator_type == "additive":
        # I_eps+(A, B) = max_{b in B} min_{a in A} max_{i} (a_i - b_i)
        vals = pf_a_reshaped - pf_b_reshaped
    elif indicator_type == "multiplicative":
        # I_eps*(A, B) = max_{b in B} min_{a in A} max_{i} (a_i / b_i)
        if np.any(pf_b <= 0):
            raise ValueError(
                "Warning: Multiplicative indicator is not suitable for objective values <= 0."
            )
        pf_b_safe = np.where(pf_b_reshaped == 0, np.finfo(float).eps, pf_b_reshaped)
        vals = pf_a_reshaped / pf_b_safe
    else:
        raise ValueError("indicator_type must be 'additive' or 'multiplicative'")

    # For each pair (a, b) find the max difference/ratio over all objectives
    max_over_objectives = np.max(vals, axis=2)  # -> (n_a, n_b)

    # For each solution b find the minimum value from all comparisons with a
    # This means finding the a that "best" dominates b
    min_over_a = np.min(max_over_objectives, axis=0)  # -> (n_b,)

    return np.max(min_over_a)
