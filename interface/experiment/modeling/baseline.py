from machinable import Interface, get, Execution
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def trial_reduction(df):
    grouped = df.groupby("key").agg(["mean", "std"])

    grouped.columns = ["_".join(col).strip() for col in grouped.columns.values]

    num_trials = df.groupby("key")["trial"].apply(lambda x: (x != -1).sum())

    result = grouped.reset_index()
    result["trial"] = result["key"].map(num_trials)

    result = result.drop(["trial_mean", "trial_std"], axis=1)

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

        for trial in range(3):
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
                    # ["~joint_model(scope=['feasiblity'])", {"dopt_params.surrogate_method_name": 'gpr'}],
                    # ["~joint_model(scope=['feasiblity'])", {"dopt_params.surrogate_method_name": 'megp'}],
                    # ["~joint_model(scope=['feasiblity'])", {"dopt_params.surrogate_method_name": 'mdspp'}],
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
                            },
                        ]
                        + version,
                    )

                    experiment.future()

    @property
    def experiments(self):
        with Execution().deferred() as staged:
            self.launch()

        return staged.executables.filter(lambda x: x.cached())

    def hv(self):
        experiments = self.experiments

        all_fronts = np.vstack([e.get_best()["y"].to_numpy() for e in experiments])
        ref_point = np.max(all_fronts, axis=0)  # nadir point
        print("Nadir reference point:", ref_point)

        d = []
        for exp in experiments:
            hv = exp.hypervolume(ref_point.tolist())
            d.append(
                {
                    "key": str(exp.version()[1]),
                    "trial": exp.predicate.get("trial", -1),
                    "hv": hv,
                    "hv_log": np.log(hv),
                }
            )

        df = pd.DataFrame(d)

        df = trial_reduction(df)
        df = normalize_column(df, "hv")

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
