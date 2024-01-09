import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
from sklearn.utils import Bunch

sns.set_context("talk")

# Define variables
# datasets
datasets = [
    "bold5000",
    # "bold5000_fold2",
    # "bold5000_fold3",
    # "bold5000_fold4",
    "forrest",
    "neuromod",
    "rsvp",
]
# n_samples = [332, 332, 332, 332, 175, 50, 360]
n_samples = [332, 175, 50, 360]
# performance metrics
metrics = ["accuracy", "balanced_accuracy"]
# features
features = ["voxels", "difumo"]
# features = ["voxels"]
# classifiers
classifiers = ["LinearSVC", "RandomForest"]
# classifiers = ["LinearSVC"]
# settings
methods = ["conventional", "stacked"]

DATA_ROOT = "/storage/store2/work/haggarwa/retreat_2023"
out_dir = os.path.join(DATA_ROOT, "plots")
os.makedirs(out_dir, exist_ok=True)

for metric in metrics:
    dfs = []
    n_subs = []
    n_classes = []
    gains = Bunch()
    gains["dataset"] = []
    gains["feature"] = []
    gains["classifier"] = []
    gains["gain"] = []
    gains["n_subs"] = []
    gains["n_classes"] = []
    gains["n_samples"] = []
    for dataset, n_sample in zip(datasets, n_samples):
        for feature in features:
            results_dir = os.path.join(
                DATA_ROOT, "results", f"{dataset}_{feature}"
            )
            for classifier in classifiers:
                pickles = glob(
                    os.path.join(results_dir, f"*{classifier}*.pkl")
                )
                for pickle in pickles:
                    df = pd.read_pickle(pickle)
                    df["dataset"] = [dataset] * df.shape[0]
                    df["feature"] = [feature] * df.shape[0]
                    acc_conventional = df[df["setting"] == "conventional"][
                        metric
                    ].values
                    acc_stacked = df[df["setting"] == "stacked"][metric].values
                    gain = acc_stacked - acc_conventional
                    gains["gain"].extend(gain)
                    gains["n_samples"].extend(
                        df[df["setting"] == "conventional"][
                            "train_size"
                        ].values
                    )
                    chance_rows = []
                    for _, row in df.iterrows():
                        if row["setting"] == "conventional":
                            chance_row = row.copy()
                            chance_row["setting"] = "chance"
                            chance_row[metric] = row[f"dummy_{metric}"]
                        else:
                            chance_row[metric] = (
                                chance_row[metric] + row[f"dummy_{metric}"]
                            ) / 2
                        chance_rows.append(chance_row)

                    n_sub = len(pickles)
                    n_class = np.unique(
                        np.concatenate(df["true"].to_list())
                    ).shape[0]
                    gains["dataset"].extend([dataset] * len(gain))
                    gains["feature"].extend([feature] * len(gain))
                    gains["classifier"].extend([classifier] * len(gain))
                    gains["n_subs"].extend([len(pickles)] * len(gain))
                    gains["n_classes"].extend([n_class] * len(gain))
                    dfs.append(df)
                    dfs.append(pd.DataFrame(chance_rows))

        n_samples.append(n_sample)
        n_subs.append(n_sub)
        n_classes.append(n_class)

    gains = pd.DataFrame(gains)
    av_gains = (
        gains.groupby(["dataset", "feature", "classifier"])
        .mean()
        .reset_index()
    )
    print(av_gains)

    df = pd.concat(dfs)
    df = df.reset_index(drop=True)
    df["Training size (%)"] = 90 - (df["left_out"] * 0.9)
    df["log_train_size"] = np.log(df["train_size"])
    df[f"log_{metric}"] = np.log(df[metric])

    for plot_type in ["feature", "classifier"]:
        if plot_type == "feature":
            selector = classifiers.copy()
            col_selector = "classifier"
        elif plot_type == "classifier":
            selector = features.copy()
            col_selector = "feature"
        for selection in selector:
            df_ = df[df[col_selector] == selection]
            av_gains_ = av_gains[av_gains[col_selector] == selection]
            fig = sns.relplot(
                data=df_,
                x="train_size",
                y=metric,
                col="dataset",
                hue="setting",
                style=plot_type,
                palette=["b", "r", "k"],
                kind="line",
                # col_wrap=2,
            )
            fig.fig.subplots_adjust(hspace=0.3)
            for i, ax in enumerate(fig.axes.flatten()):
                ax.set(
                    xscale="log",
                    # yscale="log",
                    ylim=(0, 1),
                )

                ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
                ax.ticklabel_format(style="plain")
                ax.set_title(
                    f"{datasets[i]}, {n_subs[i]} subjects,\n {n_samples[i]} samples"
                )
                values = (
                    av_gains_[av_gains_["dataset"] == datasets[i]]["gain"]
                    .values.round(2)
                    .__mul__(100)
                    .astype(int)
                )
                keys = av_gains_[av_gains_["dataset"] == datasets[i]][
                    plot_type
                ].values
                ax.text(
                    1,
                    0.9,
                    "Average gain:",
                    horizontalalignment="right",
                    verticalalignment="center",
                    transform=ax.transAxes,
                )
                ax.text(
                    1,
                    0.8,
                    f"{keys[0]}: {values[0]}%",
                    horizontalalignment="right",
                    verticalalignment="center",
                    transform=ax.transAxes,
                )
                ax.text(
                    1,
                    0.7,
                    f"{keys[1]}: {values[1]}%",
                    horizontalalignment="right",
                    verticalalignment="center",
                    transform=ax.transAxes,
                )
            fig.set_ylabels("accuracy", clear_inner=False)
            fig.set_xlabels("train size")
            sns.move_legend(
                fig,
                "lower center",
                ncol=2,
                frameon=True,
                shadow=True,
                bbox_to_anchor=(0.45, -0.3),
                title=None,
            )
            plt.savefig(
                os.path.join(
                    out_dir,
                    f"all_datasets_{metric}_{plot_type}_{selection}.png",
                ),
                bbox_inches="tight",
            )
            plt.savefig(
                os.path.join(
                    out_dir,
                    f"all_datasets_{metric}_{plot_type}_{selection}.svg",
                ),
                bbox_inches="tight",
            )

            gains_ = gains[gains[col_selector] == selection]
            fig = sns.relplot(
                data=gains_,
                x="n_samples",
                y="gain",
                col="dataset",
                # size="n_subs",
                style=plot_type,
                kind="line",
                # col_wrap=2,
                color="k",
            )
            plt.savefig(
                os.path.join(
                    out_dir,
                    f"gains_v_samples_{metric}_{plot_type}_{selection}.png",
                ),
                bbox_inches="tight",
            )
            plt.savefig(
                os.path.join(
                    out_dir,
                    f"gains_v_samples_{metric}_{plot_type}_{selection}.svg",
                ),
                bbox_inches="tight",
            )
            fig = sns.relplot(
                data=gains_,
                x="n_subs",
                y="gain",
                col="dataset",
                # size="n_samples",
                style=plot_type,
                kind="line",
                # col_wrap=2,
                color="k",
            )
            plt.savefig(
                os.path.join(
                    out_dir,
                    f"gains_v_subs_{metric}_{plot_type}_{selection}.png",
                ),
                bbox_inches="tight",
            )
            plt.savefig(
                os.path.join(
                    out_dir,
                    f"gains_v_subs_{metric}_{plot_type}_{selection}.svg",
                ),
                bbox_inches="tight",
            )
            plt.close()

    df_mean = df.groupby(["dataset", "setting", "feature", "classifier"])
    df_mean = df_mean.apply(lambda x: x)
    df_mean["setting, classifier"] = (
        df_mean["setting"] + ", " + df_mean["classifier"]
    )
    chance = df_mean[df_mean["setting"] == "chance"]
    chance = chance.groupby(["dataset"]).mean().reset_index()
    chance = chance[["dataset", metric]].set_index("dataset").to_dict()
    order = [
        "stacked, LinearSVC",
        "stacked, RandomForest",
        "conventional, LinearSVC",
        "conventional, RandomForest",
    ]
    fig = sns.catplot(
        data=df_mean,
        x=metric,
        y="feature",
        hue="setting, classifier",
        col="dataset",
        kind="bar",
        palette=[
            "r",
            "pink",
            "b",
            "skyblue",
            "k",
            "grey",
        ],
        errorbar=None,
        orient="h",
        hue_order=order,
        order=["voxels", "difumo"],
    )
    # fig.fig.subplots_adjust(hspace=0.3, wspace=0.55)
    fig.set_xlabels("average accuracy", wrap=True, clear_inner=False)
    fig.set_ylabels("features", wrap=True, clear_inner=False)
    for i, ax in enumerate(fig.axes.ravel()):
        ax.set_title(
            f"{datasets[i]}, {n_subs[i]} subjects,\n {n_samples[i]} samples"
        )
        ax.axvline(
            chance[metric][datasets[i]],
            color="k",
            linestyle="--",
            label="chance",
        )

    for i in range(4):
        ax = fig.facet_axis(0, i)
        for j in ax.containers:
            plt.bar_label(
                j,
                fmt="%.2f",
                label_type="edge",
                fontsize="x-small",
                # padding=65,
                # weight="bold",
                color="black",
            )
    fig._legend.remove()
    plt.legend(
        loc="lower center",
        ncol=3,
        frameon=True,
        shadow=True,
        bbox_to_anchor=(-1.20, -0.6),
        title="setting, classifier",
    )
    plt.savefig(
        os.path.join(
            out_dir,
            f"bench_{metric}.png",
        ),
        bbox_inches="tight",
    )
    plt.savefig(
        os.path.join(
            out_dir,
            f"bench_{metric}.svg",
        ),
        bbox_inches="tight",
    )
    plt.close()
