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


class Baseline(Interface):
    def launch(self):
        nm = "PVBC"
        protocol = [
            "interface.sopt_modeling",
            f"""~from_protocol("benchmarks/ca1_pinsky_rinzel_modeling/config/CA1_{nm}.yaml")""",
        ]

        # no surrogate
        get(protocol).future()

        for trial in range(5):
            with get("machinable.scope", {"trial": trial}):
                for version in [
                    # defaults
                    [{"dopt_params.surrogate_method_name": "gpr"}],
                    [{"dopt_params.surrogate_method_name": "megp"}],
                    [{"dopt_params.surrogate_method_name": "mdspp"}],
                    # mlp without constraints
                    ["~joint_model(scope=['objective'])"],
                    # mlp with constraints
                    ["~joint_model"],
                    # mlp constraints only with defaults
                    [
                        "~joint_model(scope=['feasiblity'])",
                        {"dopt_params.surrogate_method_name": "gpr"},
                    ],
                    [
                        "~joint_model(scope=['feasiblity'])",
                        {"dopt_params.surrogate_method_name": "megp"},
                    ],
                    [
                        "~joint_model(scope=['feasiblity'])",
                        {"dopt_params.surrogate_method_name": "mdspp"},
                    ],
                    # mlp with constraints and feasibility solving
                    ["~joint_model(feasibility_solving=True)"],
                ]:
                    experiment = get(
                        "interface.sopt_modeling",
                        [
                            f"""~from_protocol(
                                "benchmarks/ca1_pinsky_rinzel_modeling/config/CA1_PVBC.yaml"
                            )""",
                            {
                                "nodes": "10",
                                "dopt_params.n_initial": 50,
                            },
                        ]
                        + version,
                    )

                    experiment.launch()

    @property
    def experiments(self):
        with Execution().deferred() as staged:
            self.launch()

        return staged.executables

    def hv(self):
        experiments = self.experiments
        np.set_printoptions(linewidth=np.inf, suppress=True)

        # region = [0, experiments.map(lambda x: len(x.load_h5()['epochs'])).min()]
        # region = (0, 9600)
        region = None
        print(region)

        all_fronts = np.vstack(
            [e.get_best(region)["y"].to_numpy() for e in experiments[1:]]
        )
        fmax = np.max(all_fronts, axis=0).tolist()  # nadir point
        fmin = np.min(all_fronts, axis=0).tolist()
        print("Nadir reference point:", fmax)
        print("Min point", fmin)

        d = []
        for exp in experiments[1:]:
            hv = exp.hypervolume([1] * len(fmax), region, normalize=[fmin, fmax])
            d.append(
                {
                    "key": str(exp.version()[1]),
                    "trial": exp.predicate.get("trial", -1),
                    "hv": hv,
                }
            )

        df = pd.DataFrame(d)

        df = trial_reduction(df)

        print(df)

    def igd(self):
        experiments = self.experiments
        ref_front = experiments[0].get_best()["y"].to_numpy()
        d = []
        for exp in experiments[1:]:
            igd = exp.igd(ref_front)
            d.append(
                {
                    "key": object_hash(exp.context.config)[:10],
                    "name": str(exp.version()[1:]),
                    "trial": exp.predicate.get("trial", -1),
                    "igd": igd,
                }
            )

        df = pd.DataFrame(d)

        df = trial_reduction(df)

        print(df)

    # c-metric
    # experiment_names = list(experiments.keys())
    # num_experiments = len(experiment_names)

    # metric_matrix = np.zeros((num_experiments, num_experiments))

    # for i in range(num_experiments):
    #     for j in range(num_experiments):
    #         exp1 = experiments[experiment_names[i]]
    #         exp2 = experiments[experiment_names[j]]
    #         metric_matrix[i, j] = exp1.c_metric(exp2) #exp1.c_metric(exp2)

    # fig = plt.figure(figsize=(10, 8))
    # sns.heatmap(
    #     metric_matrix,
    #     xticklabels=experiment_names,
    #     yticklabels=experiment_names,
    #     annot=True,
    #     cmap="viridis",
    # )
    # plt.title("C-metric()")
    # plt.xlabel("Experiments")
    # plt.ylabel("Experiments")
