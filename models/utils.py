import numpy as np
import seaborn as sns
from matplotlib.colors import LogNorm
from matplotlib import pyplot as plt
from sklearn.utils import resample
import math
from scipy import stats


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


def epsilon_get_best(
    x,
    y,
    f,
    c,
    feasible=True,
    delete_duplicates=True,
    epsilons=None,
):
    if feasible and c is not None:
        feasible = np.argwhere(np.all(c > 0.0, axis=1)).ravel()
        if len(feasible) > 0:
            feasible = feasible.ravel()
            x = x[feasible, :]
            y = y[feasible, :]
            if f is not None:
                f = f[feasible]
            c = c[feasible, :]

    if delete_duplicates:
        from dmosopt import MOEA

        is_duplicate = MOEA.get_duplicates(y)

        x = x[~is_duplicate]
        y = y[~is_duplicate]
        if f is not None:
            f = f[~is_duplicate]
        if c is not None:
            c = c[~is_duplicate]

    if epsilons is None:
        epsilons = [1e-9] * y.shape[1]
    elif isinstance(epsilons, (int, float)):
        epsilons = [float(epsilons)] * y.shape[1]
    elif epsilons == "auto":
        # 5% of IQR
        epsilons = 0.05 * stats.iqr(y, axis=0)

    if y.shape[0] == 0:
        return x, y, f, c, epsilons

    sorter = EpsilonSort(epsilons)

    for i in range(y.shape[0]):
        sorter.sortinto(y[i], tagalong=i)

    m = np.array(sorter.tagalongs)

    best_f = None
    if f is not None:
        best_f = f[m]

    best_c = None
    if c is not None:
        best_c = c[m]

    return x[m], y[m], best_f, best_c, epsilons


class EpsilonSort:
    """
    An archive of epsilon-nondominated solutions.
    Allows auxiliary information to tag along for the sort
    process.

    The eps_sort function provides a much more convenient interface than
    the Archive class.

    ----

    Source: https://github.com/matthewjwoodruff/pareto.py/blob/master/pareto.py

    ----

    Copyright (C) 2013 Matthew Woodruff and Jon Herman.

    This script is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This script is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this script. If not, see <http://www.gnu.org/licenses/>.
    """

    def __init__(self, epsilons):
        """
        epsilons: sizes of epsilon boxes to use in the sort.  Number
                  of objectives is inferred by the number of epsilons.
        """
        self.archive = []  # objectives
        self.tagalongs = []  # tag-along data
        self.boxes = []  # remember for efficiency
        self.epsilons = [e if e != 0 and not np.isnan(e) else 1e-8 for e in epsilons]
        self.itobj = range(len(epsilons))  # infer number of objectives

    def add(self, objectives, tagalong, ebox):
        """add a solution to the archive, plus auxiliary information"""
        self.archive.append(objectives)
        self.tagalongs.append(tagalong)
        self.boxes.append(ebox)

    def remove(self, index):
        """remove a solution from the archive"""
        self.archive.pop(index)
        self.tagalongs.pop(index)
        self.boxes.pop(index)

    def sortinto(self, objectives, tagalong=None):
        """
        Sort a solution into the archive.  Add it if it's nondominated
        w.r.t current solutions.

        objectives: objectives by which to sort.  Minimization is assumed.
        tagalong:   data to preserve with the objectives.  Probably the actual
                    solution is here, the objectives having been extracted
                    and possibly transformed.  Tagalong data can be *anything*.
                    We don't inspect it, just keep a reference to it for as
                    long as the solution is in the archive, and then return
                    it in the end.
        """
        # Here's how the early loop exits in this code work:
        # break:    Stop iterating the box comparison for loop because we know
        #           the solutions are in relatively nondominated boxes.
        # continue: Start the next while loop iteration immediately (i.e.
        #           jump ahead to the comparison with the next archive member).
        # return:   The candidate solution is dominated, stop comparing it to
        #           the archive, don't add it, immediately exit the method.
        objectives = np.nan_to_num(objectives)
        ebox = [math.floor(objectives[ii] / self.epsilons[ii]) for ii in self.itobj]

        asize = len(self.archive)

        ai = -1  # ai: archive index
        while ai < asize - 1:
            ai += 1
            adominate = False  # archive dominates
            sdominate = False  # solution dominates
            nondominate = False  # neither dominates

            abox = self.boxes[ai]

            for oo in self.itobj:
                if abox[oo] < ebox[oo]:
                    adominate = True
                    if sdominate:  # nondomination
                        nondominate = True
                        break  # for
                elif abox[oo] > ebox[oo]:
                    sdominate = True
                    if adominate:  # nondomination
                        nondominate = True
                        break  # for

            if nondominate:
                continue  # while
            if adominate:  # candidate solution was dominated
                return
            if sdominate:  # candidate solution dominated archive solution
                self.remove(ai)
                ai -= 1
                asize -= 1
                continue  # while

            # solutions are in the same box
            aobj = self.archive[ai]
            corner = [ebox[ii] * self.epsilons[ii] for ii in self.itobj]
            sdist = sum([(objectives[ii] - corner[ii]) ** 2 for ii in self.itobj])
            adist = sum([(aobj[ii] - corner[ii]) ** 2 for ii in self.itobj])
            if adist < sdist:  # archive dominates
                return
            else:  # solution dominates
                self.remove(ai)
                ai -= 1
                asize -= 1
                # Need a continue here if we ever reorder the while loop.
                continue  # while

        # if you get here, then no archive solution has dominated this one
        self.add(objectives, tagalong, ebox)
