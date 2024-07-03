import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
from sklearn.utils import Bunch
from tqdm import tqdm

sns.set_context("talk", font_scale=1.2)
# datasets
datasets = [
    "neuromod",
    "forrest",
    "bold5000",
    "rsvp",
    # "nsd",
    "aomic_anticipation",
]
# Camelized dataset names
fixed_datasets = {
    "neuromod": "Neuromod",
    "forrest": "Forrest",
    "bold5000": "BOLD5000",
    "rsvp": "RSVP-IBC",
    # "nsd": "NSD",
    "aomic_anticipation": "AOMIC",
}
# n_samples = [50, 175, 332, 360, 4848, 61]
n_samples = [50, 175, 332, 360, 61]
# performance metrics
# metrics = ["accuracy", "balanced_accuracy"]
metrics = ["balanced_accuracy"]
# features
features = ["voxels", "difumo"]
# fixed feature names
fixed_features = {"voxels": "Voxels", "difumo": "DiFuMo"}
# classifiers
classifiers = ["LinearSVC", "RandomForest"]
# settings
methods = ["conventional", "stacked"]
# fixed method names
fixed_methods = {
    "conventional": "Conventional",
    "stacked": "Ensemble",
    "chance": "Chance",
}
# lineplot types
plot_types = ["Classifier"]
# plot_types = ["Feature", "Classifier"]

DATA_ROOT = "."
out_dir = os.path.join(DATA_ROOT, "plots_copy_old")
os.makedirs(out_dir, exist_ok=True)

### Calculate average gains
for metric in metrics:
    dfs = []
    n_subs = []
    n_classes = []
    gains = Bunch()
    gains["dataset"] = []
    gains["Feature"] = []
    gains["Classifier"] = []
    gains["gain"] = []
    gains["n_subs"] = []
    gains["n_classes"] = []
    gains["n_samples"] = []

    print(f"Calculating gains for {metric}...")

    for dataset, n_sample in tqdm(
        zip(datasets, n_samples), total=len(datasets)
    ):
        for feature in features:
            results_dir = os.path.join(
                DATA_ROOT, "results_l2", f"{dataset}_{feature}"
            )
            for classifier in classifiers:
                pickles = glob(
                    os.path.join(results_dir, f"*{classifier}*.pkl")
                )
                for pickle in pickles:
                    df = pd.read_pickle(pickle)

                    df["dataset"] = [dataset] * df.shape[0]
                    df["Feature"] = [feature] * df.shape[0]

                    # fix column names
                    df.rename(
                        {
                            "setting": "Setting",
                            "classifier": "Classifier",
                        },
                        axis="columns",
                        inplace=True,
                    )

                    # fix names
                    df["Setting"] = df["Setting"].map(fixed_methods)
                    df["Feature"] = df["Feature"].map(fixed_features)

                    acc_conventional = df[df["Setting"] == "Conventional"][
                        metric
                    ].values
                    acc_stacked = df[df["Setting"] == "Ensemble"][
                        metric
                    ].values
                    gain = acc_stacked - acc_conventional
                    gains["gain"].extend(gain)
                    gains["n_samples"].extend(
                        df[df["Setting"] == "Conventional"][
                            "train_size"
                        ].values
                    )
                    chance_rows = []
                    for _, row in df.iterrows():
                        if row["Setting"] == "Conventional":
                            chance_row = row.copy()
                            chance_row["Setting"] = "Chance"
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
                    gains["Feature"].extend([feature] * len(gain))
                    gains["Classifier"].extend([classifier] * len(gain))
                    gains["n_subs"].extend([len(pickles)] * len(gain))
                    gains["n_classes"].extend([n_class] * len(gain))
                    dfs.append(df)
                    dfs.append(pd.DataFrame(chance_rows))

        n_samples.append(n_sample)
        n_subs.append(n_sub)
        n_classes.append(n_class)

    gains = pd.DataFrame(gains)
    av_gains = (
        gains.groupby(["dataset", "Feature", "Classifier"])
        .mean()
        .reset_index()
    )
    av_gains["Feature"] = av_gains["Feature"].map(fixed_features)
    print(av_gains)

    ### start plotting
    df = pd.concat(dfs)
    df = df.reset_index(drop=True)

    ### lineplots over varying training size
    for plot_type in plot_types:
        print(f"Plotting {metric} {plot_type} lineplots...")
        if plot_type == "Feature":
            selector = classifiers.copy()
            col_selector = "Classifier"
        elif plot_type == "Classifier":
            selector = features.copy()
            col_selector = "Feature"
        for selection in tqdm(selector):
            if plot_type == "Classifier":
                selection = fixed_features[selection]
            df_ = df[df[col_selector] == selection]
            av_gains_ = av_gains[av_gains[col_selector] == selection]
            fig = sns.relplot(
                data=df_,
                x="train_size",
                y=metric,
                col="dataset",
                hue="Setting",
                style=plot_type,
                palette=["b", "r", "k"],
                kind="line",
                facet_kws={
                    "sharey": False,
                    "sharex": False,
                },
                # col_wrap=2,
            )
            fig.fig.subplots_adjust(hspace=0.3)
            for i, ax in enumerate(fig.axes.flatten()):
                # ax.set(
                #     xscale="log",
                #     # yscale="log",
                #     # ylim=(0, 1),
                #     # xlim=(0, 5000),
                # )
                ax.set_xscale("log", base=2)
                if n_samples[i] > 100:
                    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
                    ax.ticklabel_format(style="plain")
                    fmt = ax.get_xaxis().get_major_formatter()
                else:
                    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
                    fmt = ax.get_xaxis().get_major_formatter()
                    fmt.set_scientific(False)
                    print(fmt.__dict__)

                title = f"{fixed_datasets[datasets[i]]}, {n_subs[i]} subjects,\n {n_samples[i]} samples"
                ax.text(
                    0.5,
                    1.5,
                    title,
                    transform=ax.transAxes,
                    horizontalalignment="center",
                    verticalalignment="center",
                )
                stats = "Average gain:\n"
                values = (
                    av_gains_[av_gains_["dataset"] == datasets[i]]["gain"]
                    .values.round(2)
                    .__mul__(100)
                    .astype(int)
                )
                keys = av_gains_[av_gains_["dataset"] == datasets[i]][
                    plot_type
                ].values

                ax.set_title("")
                for i, (key, value) in enumerate(zip(keys, values)):
                    stats += f"{key}: {value}%\n"
                ax.set_title(stats, loc="right", fontsize="small")
            fig.set_ylabels("Accuracy", clear_inner=False)
            fig.set_xlabels("Training size")
            sns.move_legend(
                fig,
                "lower center",
                ncol=2,
                frameon=True,
                shadow=True,
                bbox_to_anchor=(0.45, -0.4),
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
            plt.close()

    print(f"Plotting {metric} barplots...")
    ### mean barplots
    df_mean = df.groupby(["dataset", "Setting", "Feature", "Classifier"])
    df_mean = df_mean.apply(lambda x: x)
    df_mean["Setting, Classifier"] = (
        df_mean["Setting"] + ", " + df_mean["Classifier"]
    )
    chance = df_mean[df_mean["Setting"] == "Chance"]
    chance = chance.groupby(["dataset"]).mean().reset_index()
    chance = chance[["dataset", metric]].set_index("dataset").to_dict()
    order = [
        "Ensemble, MLP",
        "Ensemble, LinearSVC",
        "Ensemble, RandomForest",
        "Conventional, MLP",
        "Conventional, LinearSVC",
        "Conventional, RandomForest",
    ]
    fig = sns.catplot(
        data=df_mean,
        x=metric,
        y="Feature",
        hue="Setting, Classifier",
        col="dataset",
        kind="bar",
        palette="tab20c",
        errorbar="ci",
        orient="h",
        hue_order=order,
        order=["Voxels", "DiFuMo"],
    )
    # fig.fig.subplots_adjust(hspace=0.3, wspace=0.55)
    fig.set_xlabels("Average accuracy", wrap=True, clear_inner=False)
    fig.set_ylabels("Features", wrap=True, clear_inner=False)
    for i, ax in enumerate(fig.axes.ravel()):
        ax.set_title(
            f"{fixed_datasets[datasets[i]]}, {n_subs[i]} subjects,\n {n_samples[i]} samples"
        )
        ax.axvline(
            chance[metric][datasets[i]],
            color="k",
            linestyle="--",
            label="Chance",
        )

    for i in range(len(datasets)):
        ax = fig.facet_axis(0, i)
        for j in ax.containers:
            plt.bar_label(
                j,
                fmt="%.2f",
                label_type="edge",
                fontsize="x-small",
                padding=10,
                # weight="bold",
                color="black",
            )
    fig._legend.remove()
    plt.legend(
        loc="lower center",
        ncol=3,
        frameon=True,
        shadow=True,
        bbox_to_anchor=(-1.8, -0.8),
        title="Setting, Classifier",
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
