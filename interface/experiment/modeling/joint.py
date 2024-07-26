from machinable import Interface, get, Execution
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from machinable.utils import object_hash


def trial_reduction(df):
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    non_numeric_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

    if "key" in non_numeric_cols:
        non_numeric_cols.remove("key")
    if "trial" in numeric_cols:
        numeric_cols.remove("trial")

    grouped = df.groupby("key")[numeric_cols].agg(["mean", "std"])
    grouped.columns = ["_".join(col).strip() for col in grouped.columns.values]
    num_trials = df.groupby("key")["trial"].apply(lambda x: (x != -1).sum())

    result = grouped.reset_index()
    result["trial"] = result["key"].map(num_trials)

    if "trial_mean" in result.columns and "trial_std" in result.columns:
        result = result.drop(["trial_mean", "trial_std"], axis=1)

    non_numeric_df = df[non_numeric_cols + ["key"]].drop_duplicates(subset=["key"])
    result = result.merge(non_numeric_df, on="key", how="left")

    return result


def normalize_column(df, name):
    mean_col = f"{name}_mean"
    std_col = f"{name}_std"

    min_val = df[mean_col].min()
    max_val = df[mean_col].max()
    df[f"{name}_mean_norm"] = (df[mean_col] - min_val) / (max_val - min_val)
    df[f"{name}_mean_std"] = df[std_col] / (max_val - min_val)

    return df


class Joint(Interface):

    def launch(self):
        for trial in range(1):
            with get("machinable.scope", {"trial": trial, "cohort": "joint3"}):
                for nm in [
                    "SCA",
                    "IVY",
                    "PVBC",
                    "CCKBC",
                    "AAC",
                    "BS",
                    "OLM",
                    "NGFC",
                    "IS",
                ]:
                    protocol = [
                        "interface.sopt_modeling",
                        f"""~from_protocol("benchmarks/ca1_pinsky_rinzel_modeling/config/CA1_{nm}.yaml")""",
                    ]

                    # no surrogate
                    # get(protocol).launch()

                    for version in [
                        # standard surrogates
                        [
                            {
                                "dopt_params.surrogate_method_name": "gpr",
                                "dopt_params.surrogate_method_kwargs": {
                                    "top_k": 250,
                                },
                            }
                        ],
                        [
                            {
                                "dopt_params.surrogate_method_name": "megp",
                                "dopt_params.surrogate_method_kwargs": {
                                    "top_k": 250,
                                },
                            }
                        ],
                        # mlp standalone
                        ["~joint_model(mode='o')"],
                        # mlp with constraints
                        ["~joint_model(mode='c+o')"],
                        # mlp with feasibility solving
                        ["~joint_model(feasibility_solving=True)"],
                        # sensitivity
                        ["~joint_model(mode='o', sensitivity=True)"],
                        ["~joint_model(mode='c+o', sensitivity=True)"],
                        ["~joint_model(feasibility_solving=True, sensitivity=True)"],
                        ["~joint_model(mode='o', sensitivity='cross_check')"],
                        ["~joint_model(mode='c+o', sensitivity='cross_check')"],
                        [
                            "~joint_model(feasibility_solving=True, sensitivity='cross_check')"
                        ],
                        # todo: dynamic sampling
                    ]:
                        experiment = get(
                            protocol,
                            [
                                {
                                    "nodes": "96",
                                    # "dopt_params.n_initial": 500,# 250, 100, 50
                                },
                            ]
                            + version,
                        )
                        # print(experiment.get_best()['y'])
                        experiment.launch()

    def inspect(self):
        region = 10000000
        for experiment in self.components:
            print(experiment.label())
            print(experiment.local_directory())
            print(experiment.output_filepath)
            print(experiment.get_best(sort_by="-np.max(y, axis=1)")["y"])
            eps = experiment.load_h5()["epochs"]
            if eps < region:
                region = eps
            progress = str(len(eps))
            progress += " / "
            progress += str(experiment.num_evals_total)
            print(progress)
            print("-----------")
        print("Region: ", [0, region])

    def hv(self):
        experiments = self.components
        np.set_printoptions(linewidth=np.inf, suppress=True)

        region = None
        all_fronts = np.vstack(
            [e.get_best(region)["y"].to_numpy() for e in experiments]
        )
        fmax = np.max(all_fronts, axis=0).tolist()  # nadir point
        fmin = np.min(all_fronts, axis=0).tolist()
        print("Nadir reference point:", fmax)
        print("Min point", fmin)

        d = []
        for exp in experiments:
            hv = exp.hypervolume([1] * len(fmax), region, normalize=[fmin, fmax])
            d.append(
                {
                    "key": object_hash(exp.context.config)[:10],
                    "name": exp.label(),
                    "trial": exp.predicate.get("trial", -1),
                    "hv": hv,
                    "hv_r": (hv / 1.0) * 100,
                }
            )

        df = pd.DataFrame(d)

        df = trial_reduction(df)

        print(df.sort_values("hv_mean"))

        # df.to_csv(self.local_directory('hv.csv'))

    def igd(self):
        experiments = self.components
        ref_front = np.zeros_like(experiments[0].get_best()["y"].to_numpy())
        print(np.sort(ref_front, axis=0))
        d = []
        for idx, exp in enumerate(experiments):  # [1:]):
            print(exp)
            pf = exp.get_best(region=(0, 3000))["y"].to_numpy()
            if len(pf) == 0:
                continue
            igd = exp.igd(ref_front, pf=pf)
            d.append(
                {
                    "key": object_hash(exp.context.config)[:10],
                    "name": str(idx) + "::" + exp.label(),
                    "trial": exp.predicate.get("trial", -1),
                    "igd": igd,
                }
            )

        df = pd.DataFrame(d)

        df = trial_reduction(df)

        print(df.sort_values("igd_mean"))

        # df.to_csv(self.local_directory('igd.csv'))

    def c_metric(self):
        experiments = self.components
        region = None

        ref_front = experiments[0].get_best(region=region)["y"].to_numpy()
        d = []
        for idx, exp in enumerate(experiments[1:]):
            print(exp, idx, "/", len(experiments))
            d.append(
                {
                    "key": object_hash(exp.context.config)[:10],
                    "name": str(idx) + "::" + exp.label(),
                    "trial": exp.predicate.get("trial", -1),
                    "c": exp.c_metric(ref_front),
                }
            )

        df = pd.DataFrame(d)

        print(df)
        df = trial_reduction(df)

        print(df.sort_values("c_mean"))

        # df.to_csv(self.local_directory('cmetric.csv'))

    def leaderboard(self):
        d = []
        for idx, exp in enumerate(self.components):
            if not os.path.exists(exp.output_filepath):
                continue
            print(exp, idx, "/", len(self.components))
            print(exp.output_filepath)
            total = len(exp.load_h5()["objectives"].to_numpy())
            pf = exp.get_best(sort_by="-np.max(y, axis=1)", epsilon=None)[
                "y"
            ].to_numpy()
            d.append(
                {
                    "name": exp.label(),
                    "samples": str(exp.num_initial_samples) + " / " + str(total),
                    "y*": np.round(pf[-1], 2) if len(pf) > 0 else 0,
                    "len(pf)": len(pf),
                }
            )

        df = pd.DataFrame(d)

        print(df)

        # df.to_csv(self.local_directory('leaderboard.csv')

    def postmortem(self):
        idx = 2  # 14 #2       experiment = self.components[idx]

        print(experiment.label())

        data = experiment.load_h5()
        x = data["parameters"].to_numpy()
        y = data["objectives"].to_numpy()
        yC = (data["constraints"].to_numpy() > 0).astype(int)

        region = slice(0, 6000)

        if True:
            model = experiment.get_model("mlp", mode="c+o")

            epochs = 1
            # epochs = model.autoepoch(
            #         x[region],
            #         y[region],
            #         yC[region],
            #         #epochs=100,
            #         verbose=2,
            #     )
            # raise ValueError(epochs)
        else:
            model = experiment.get_model("gpr", top_k=100)
            epochs = None

        print(epochs, "asdfasd")
        # return
        model.autofit(
            x[region],
            y[region],
            yC[region],
            epochs=epochs,
            verbose=2,
        )

        test_region = slice(6000, 6500)
        # test_region = slice(5500, 6001)
        r = model.autoeval(
            x[test_region],
            y[test_region],
            yC[test_region],
        )

        r.update(
            {
                k + "_train": v
                for k, v in model.autoeval(
                    x[region],
                    y[region],
                    yC[region],
                ).items()
            }
        )

        from pprint import pprint

        pprint(r)

        return

        for r, m in enumerate(["mlp", "gpr", "megp"]):
            evals = []
            for e in range(np.max(np.unique(data["epochs"])) - 1):
                if (
                    len(y[data["epochs"] == e]) < 4
                    or len(y[data["epochs"] == e + 1]) < 4
                ):
                    continue

                tp = {"mode": "c+o"} if m == "mlp" else {"top_k": 100}
                model = experiment.get_model(m, **tp)

                model.autofit(
                    x[data["epochs"] <= e],
                    y[data["epochs"] <= e],
                    yC[data["epochs"] <= e],
                    epochs=2500,
                    verbose=0,
                )
                evals.append(
                    model.autoeval(
                        x[data["epochs"] == e + 1],
                        y[data["epochs"] == e + 1],
                        yC[data["epochs"] == e + 1],
                    )
                )
                evals[-1].update(
                    {
                        k + "_train": v
                        for k, v in model.autoeval(
                            x[data["epochs"] == e],
                            y[data["epochs"] == e],
                            yC[data["epochs"] == e],
                        ).items()
                    }
                )
                print(evals[-1])

            df = pd.DataFrame(evals)

            print(df)

    def clear_cache(self):
        for c in self.components:
            for fn in os.listdir(c.local_directory()):
                if not fn.startswith(".cachable_") and not fn.endswith(".p"):
                    continue
                print("Clearing ", c.label(), c.local_directory())
                os.remove(c.local_directory(fn))

    def colormap_bars(self):
        from nds import ndomsort
        from collections import defaultdict

        # populations = self.components.map(lambda x: x.label().split('_')[1]).unique()
        populations = ["SCA", "IVY", "PVBC", "CCKBC", "AAC", "BS", "OLM", "NGFC", "IS"]

        fig, ax = plt.subplots()

        names = None
        bars = defaultdict(list)
        for population in populations:
            experiments = self.components.filter(
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
            front_indices = np.array(
                ndomsort.non_domin_sort(seq, only_front_indices=True)
            )
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

        fig.savefig("plot.png")

    def colormap_pie(self):
        from nds import ndomsort
        from models.utils import EpsilonSort
        import matplotlib as mpl
        from matplotlib.lines import Line2D

        # populations = self.components.map(lambda x: x.label().split('_')[1]).unique()
        populations = ["SCA", "IVY", "PVBC", "CCKBC", "AAC", "BS", "OLM", "NGFC", "IS"]

        fig, axs = plt.subplots(3, 3, figsize=(25, 25))

        for ax, population in zip(axs.ravel(), populations):

            experiments = self.components.filter(lambda x: population in x.label())

            num_y = experiments[0].num_objectives

            # sorter = EpsilonSort([1e-8]*num_y)
            labels = []
            colors = []
            lines = []
            pf_sizes = {}

            cmap = mpl.colormaps["plasma"]
            cs = cmap(np.linspace(0, 1, len(experiments)))

            seq = []

            for e, experiment in enumerate(experiments):
                print(e + 1, "/", len(experiments))
                pf = experiment.get_best(epsilon=None)["y"].to_numpy()
                l = experiment.label().split("::")[
                    -1
                ]  # + str(experiment.config.dopt_params.get('surrogate_custom_training_kwargs', {}).get('outlier_threshold', '?'))
                pf_sizes[l] = len(pf)
                colors.append(cs[e])
                lines.append(Line2D([0], [0], color=colors[e], lw=4))
                for i in range(pf.shape[0]):
                    seq.append(list(pf[i]))
                    labels.append(l)

            front_indices = np.array(
                ndomsort.non_domin_sort(seq, only_front_indices=True)
            )
            labels = np.array(labels)

            print(front_indices, labels)

            if False:
                unique_fronts = np.sort(np.unique(front_indices))
                unique_labels = np.unique(labels)

                counts = np.zeros((len(unique_fronts), len(unique_labels)))

                for i, front in enumerate(unique_fronts):
                    mask = front_indices == front
                    for j, label in enumerate(unique_labels):
                        counts[i, j] = np.sum((labels == label) & mask)

                fig, ax = plt.subplots(figsize=(10, 6))

                bottom = np.zeros(len(unique_fronts))
                for j, label in enumerate(unique_labels):
                    ax.bar(unique_fronts, counts[:, j], bottom=bottom, label=label)
                    bottom += counts[:, j]

                ax.set_xlabel("Front")
                ax.set_ylabel("Count")
                ax.set_title("Composition of Fronts by Method")
                ax.legend()

                # Set x-axis ticks to integers
                ax.set_xticks(unique_fronts)
                ax.set_xticklabels(unique_fronts.astype(int))

                # Add total count labels on top of each bar
                for i, front in enumerate(unique_fronts):
                    total = np.sum(counts[i])
                    ax.text(front, total, f"{total:.0f}", ha="center", va="bottom")
            else:
                pareto_mask = front_indices == 0
                pareto_labels = labels[pareto_mask]
                unique_labels, counts = np.unique(pareto_labels, return_counts=True)

                ax.pie(counts, labels=unique_labels, autopct="%1.1f%%", startangle=90)
                ax.axis("equal")

                ax.set_title(population)

            # break

        plt.tight_layout()

        fig.savefig("plot.png")

    def colormap(self):
        from dmosopt.MOEA import orderMO
        from matplotlib.lines import Line2D
        import matplotlib as mpl

        experiments = self.components.filter(lambda x: "IVY" in x.label())

        labels = []
        colors = []
        lines = []

        cmap = mpl.colormaps["plasma"]
        cs = cmap(np.linspace(0, 1, len(experiments)))

        data = []

        for e, experiment in enumerate(experiments):
            print(e + 1, "/", len(experiments))
            pf = experiment.get_best()["y"].to_numpy()

            print(pf, experiment.label())

            for i in range(pf.shape[0]):
                data.append([e] + list(pf[i]))

            labels.append(experiment.label())
            colors.append(cs[e])
            lines.append(Line2D([0], [0], color=colors[e], lw=4))

        data = np.array(data)

        s, _, _ = orderMO([], data[:, 1:])

        sd = data[s, :]

        fig, ax = plt.subplots(figsize=(8, 4))

        for i, value in enumerate(sd):
            print(value)
            ax.axhline(y=-i, xmax=1, color=colors[int(value[0])], linewidth=1)

        ax.legend(lines, labels)

        # ax.set_xlim([0,2])
        ax.set_yticks([])
        # ax.set_yticklabels(data)

        print("best", sd[0, :])
        print("worst", sd[-1, :])

        plt.tight_layout()

        fig.savefig("plot.png")

    def constraint_curve(self):
        populations = ["SCA", "IVY", "PVBC", "CCKBC", "AAC", "BS", "OLM", "NGFC", "IS"]

        fig, axs = plt.subplots(3, 3, figsize=(16, 12))

        for ax, population in zip(axs.ravel(), populations):

            experiments = self.components.filter(
                lambda x: population in x.label() and x.cached()
            )

            for experiment in experiments:
                C = experiment.load_h5()["constraints"].to_numpy()
                feasible = np.mean(C > 0, axis=1)
                # average in chunks of a 100
                chunk_size = 500
                start = 0  # experiment.num_initial_samples
                num = len(feasible) - start
                cutoff = num - (num % chunk_size)
                end = start + cutoff
                reduce = feasible[start:end].reshape(-1, chunk_size).mean(axis=1)

                ax.plot(
                    np.arange(start, end, chunk_size), reduce, label=experiment.label()
                )
                ax.vlines(
                    experiment.num_initial_samples,
                    np.min(reduce),
                    np.max(reduce),
                    color="black",
                )

            ax.title.set_text(population)

            ax.legend()
            # ax.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)

        plt.tight_layout()

        fig.savefig("plot.png")

        return fig
