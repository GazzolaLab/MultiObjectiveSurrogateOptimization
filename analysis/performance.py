# %%
import os
from machinable import get, Component
import pickle
import numpy as np

get("machinable.index", os.environ["STORAGE"]).__enter__()

experiments = {
    "baseline": "049af43f0bf2-0008-82e7-0961-3727a560",
    "t90k": "b567456b327c-0008-1e77-c295-ec96a560",
    "t1": "004e67e5f21f-0008-2627-0535-5f16a560",
    "t2": "1dfb14537500-0008-2f97-9075-5f16a560",
    "t3": "d9f607275525-0008-48a7-2995-5f16a560",
    "mlp": "c3e82bd82d43-0008-0fd7-a037-094db560",
}

# %%

def dominates(row, candidate_row):
    return all(r <= c for r, c in zip(row, candidate_row)) and any(r < c for r, c in zip(row, candidate_row))

def set_coverage(a, b):
    """
    Calculate the set coverage of A over B, i.e. C(A, B),
    which is the fraction of solutions in B that are dominated by at least one solution in A.
    """
    coverage_count = 0
    for candidate in b:
        for solution in a:
            if dominates(solution, candidate):
                coverage_count += 1
                break
    return coverage_count / len(b)


def cross_coverage(setA, setB):
    return set_coverage(setA, setB), set_coverage(setB, setA)

# %%

if not os.path.isfile("results/performance.p"):
    baseline = Component.find_by_id(experiments["baseline"])
    mlp = Component.find_by_id(experiments["mlp"])
    mlp_best = mlp.get_best(region=slice(25000, 50000))

    px, py = [],[]
    for i in range(0, 400000, 25000):
        print(i)
        px.append(i)
        baseline_best = baseline.get_best(region=slice(i, i+25000))
        py.append(cross_coverage(mlp_best["y"], baseline_best["y"]))

    with open('results/performance.p', "wb") as f:
        pickle.dump(
            {
                "px": px,
                "py": py,
            },
            f,
        )
    
    # todo: hv
    # global_ref = []
    # gpf = []
    # for k, v in results.items():
    #     global_ref.append(np.nanmax(v['best_y'], axis=0))
    #     gpf.append(v['best_y'])
        
    # gpf = np.concatenate(gpf, axis=0)
    # global_ref = np.nanmax(global_ref, axis=0)
else:
    with open('results/performance.p', "rb") as f:
        data = pickle.load(f)
        px = data["px"]
        py = data["py"]

# %%

import matplotlib.pyplot as plt

plt.plot([ppx for ppx in px], [ppy[0] for ppy in py], label="C(grad, baseline)")

plt.xlabel("Number of evaluations")
plt.ylabel("C(feasibility_grad, baseline)")
plt.hlines(
        0.5, 0, 400000, linestyles="dashed", colors="green", label="Feasible"
)
plt.vlines(
    50000,
    0,
    1,
    linestyles="dashed",
    colors="orange",
    label="feasibility_grad optimization steps",
)
plt.legend()
plt.show()
# %%
