from nds import ndomsort
from collections import defaultdict

# populations = self.components.map(lambda x: x.label().split('_')[1]).unique()
populations = [
    "SCA",
    "IVY",
    "PVBC",
    "CCKBC",
    "AAC",
    "BS",
    "OLM",
]  # "NGFC", "IS"]

fig, ax = plt.subplots()

names = None
bars = defaultdict(list)
for population in populations:
    experiments = surrogates.components.filter(
        lambda x: population in x.label() and x.cached()
    )

    pf_sizes = {}
    labels = []
    seq = []
    for e, experiment in enumerate(experiments):
        print(e + 1, "/", len(experiments))
        h5 = experiment.load_h5()
        # pf = h5['objectives'].to_numpy()[h5['epochs'] == 2]
        pf = experiment.get_best(epsilon=None)["y"].to_numpy()
        lbl = experiment.label().split("::")[-1]
        pf_sizes[lbl] = len(pf)
        for i in range(pf.shape[0]):
            if np.any(np.isnan(pf[i])):
                print(pf[i], "nana")
                continue
            seq.append(list(pf[i]))
            labels.append(lbl)
    if len(seq) == 0:
        continue

    print(population)
    front_indices = np.array(ndomsort.non_domin_sort(seq, only_front_indices=True))
    labels = np.array(labels)

    if names is None:
        names = sorted(np.unique(labels))

    pareto_mask = front_indices == 0
    pareto_labels = labels[pareto_mask]
    num_total = len(pareto_labels)
    unique_labels, counts = np.unique(pareto_labels, return_counts=True)

    # alternative:
    if True:
        # percentage of superfront
        percentages = counts / num_total * 100
    else:
        # how much of the each pareto front makes it into the superfront?
        percentages = []
        for ul, c in zip(unique_labels, counts):
            percentages.append(c / pf_sizes[ul] * 100)
        percentages = np.array(percentages)

    for n in names:
        flag = False
        for l, p in zip(unique_labels, percentages):
            if n == l:
                flag = True
                bars[n].append(p)
                break
        if not flag:
            # missing
            bars[n].append(0)

offset = 0
for name in names:
    bar = bars[name]
    ax.barh(
        populations,
        bar,
        align="center",
        left=offset,
        height=0.75,
        label=name.replace("O:", "").replace("/C:-/S:-", "").replace("[-]", ""),
    )
    offset += np.array(bar)  # .max(axis=0)

ax.legend(
    bbox_to_anchor=(0, 1.02, 1, 0.2),
    loc="lower left",
    mode="expand",
    borderaxespad=0,
    ncol=len(names),
)

plt.tight_layout()
