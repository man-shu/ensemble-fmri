"""
This script adds additional accuracy bars and lines from the GCN classifier,
as obtained from the `compare_with_gcn.py` script. The plots are saved in the 
`plots/gcn`.

Description of the plots:
1. fig2_GCN: barplots of average accuracy for each dataset, feature, classifier, 
and setting.
2. extra lineplots of accuracy vs. number of samples per class for each
dataset, feature, classifier, and setting.
"""

import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
from tqdm import tqdm


sns.set_context("talk", font_scale=1.2)
# datasets
datasets = [
    "neuromod",
    "aomic_anticipation",
    "forrest",
    "bold",
    "rsvp",
]
# Camelized dataset names
fixed_datasets = {
    "neuromod": "Neuromod",
    "forrest": "Forrest",
    "bold": "BOLD5000",
    "rsvp": "RSVP-IBC",
    "aomic_anticipation": "AOMIC",
}
n_samples = [50, 61, 175, 332, 360]
# performance metrics
metrics = ["balanced_accuracy"]
# features
features = ["voxels", "difumo"]
# fixed feature names
fixed_features = {"voxels": "Voxels", "difumo": "DiFuMo"}
# classifiers
classifiers = ["MLP", "LinearSVC", "RandomForest", "GNN"]
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

DATA_ROOT = "."
out_dir = os.path.join(DATA_ROOT, "plots", "gcn")
os.makedirs(out_dir, exist_ok=True)

### Calculate average gains
for metric in metrics:
    dfs = []
    n_subs = []
    n_classes = []

    for order, (dataset, n_sample) in tqdm(
        enumerate(zip(datasets, n_samples)), total=len(datasets)
    ):
        for feature in features:
            results_dir = os.path.join(
                DATA_ROOT,
                "results",
                f"{dataset}_{feature}",
            )
            for classifier in classifiers:
                pickles = glob(
                    os.path.join(results_dir, f"*{classifier}*.pkl")
                )
                for pickle in pickles:
                    df = pd.read_pickle(pickle)

                    df["dataset"] = [dataset] * df.shape[0]
                    df["Feature"] = [feature] * df.shape[0]
                    df["Order"] = [order] * df.shape[0]

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

                    # n subjects and classes
                    n_sub = len(pickles)
                    n_class = np.unique(
                        np.concatenate(df["true"].to_list())
                    ).shape[0]

                    # add n_class and n_samples_per_class columns
                    df["n_classes"] = [n_class] * df.shape[0]
                    df["n_samples_per_class"] = (
                        df["train_size"] / df["n_classes"]
                    )

                    dfs.append(df)

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

                    dfs.append(pd.DataFrame(chance_rows))

        n_samples.append(n_sample)
        n_subs.append(n_sub)
        n_classes.append(n_class)

    ### start plotting
    df = pd.concat(dfs)
    df = df.reset_index(drop=True)
    df[metric] = df[metric] * 100

    print(f"Plotting {metric} barplots...")

    ### mean barplots
    df_mean = df.groupby(["dataset", "Setting", "Feature", "Classifier"])
    df_mean = df_mean.apply(lambda x: x)
    df_mean["Setting, Classifier"] = (
        df_mean["Setting"] + ", " + df_mean["Classifier"]
    )
    chance = df_mean[df_mean["Setting"] == "Chance"]
    chance = chance[["dataset", metric]].set_index("dataset").to_dict()
    order = [
        "Ensemble, MLP",
        "Ensemble, LinearSVC",
        "Ensemble, RandomForest",
        "Conventional, MLP",
        "Conventional, LinearSVC",
        "Conventional, RandomForest",
        "Conventional, GNN",
    ]
    colors = sns.color_palette("tab20c")
    ensemble_colors = colors[:3]
    conventional_colors = colors[4:7]
    gnn_color = [colors[8]]
    colors = ensemble_colors + conventional_colors + gnn_color

    fig = sns.catplot(
        data=df_mean,
        x=metric,
        y="Feature",
        hue="Setting, Classifier",
        col="dataset",
        kind="bar",
        palette=colors,
        errorbar="ci",
        orient="h",
        hue_order=order,
        order=["Voxels", "DiFuMo"],
        col_wrap=3,
        col_order=[
            "neuromod",
            "aomic_anticipation",
            "forrest",
            "bold",
            "rsvp",
        ],
    )
    fig.fig.subplots_adjust(wspace=0.2)
    fig.set_xlabels("Average accuracy (%)", wrap=True, clear_inner=False)
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
                fmt="%.1f",
                label_type="edge",
                fontsize="x-small",
                padding=10,
                color="black",
            )

    for i in range(len(datasets)):
        ax = fig.facet_axis(0, i)
        values = []
        indices = []
        for idx, child in enumerate(ax.__dict__["_children"]):
            if isinstance(child, plt.Text):
                values.append(float(child.get_text()))
                indices.append(idx)
        values = np.array(values)
        indices = np.array(indices)
        max_idx = indices[np.argmax(values)]
        ax.__dict__["_children"][max_idx].set_color("r")
        ax.__dict__["_children"][max_idx].set_weight("bold")

    fig._legend.remove()
    plt.legend(
        loc="lower right",
        ncol=1,
        frameon=True,
        shadow=True,
        bbox_to_anchor=(2.5, 0),
        title="Setting, Classifier",
        fontsize="small",
    )
    plt.savefig(
        os.path.join(
            out_dir,
            "fig2_GCN.png",
        ),
        bbox_inches="tight",
    )
    plt.savefig(
        os.path.join(
            out_dir,
            "fig2_GCN.svg",
        ),
        bbox_inches="tight",
    )
    plt.close()

    ### lineplots over varying training size
    colors = sns.color_palette("tab10")
    ensemble_colors = colors[0]
    conventional_colors = colors[1]
    gnn_color = colors[2]
    colors = [ensemble_colors, conventional_colors, gnn_color, "k"]
    for plot_type in plot_types:
        print(f"Plotting {metric} {plot_type} lineplots...")
        if plot_type == "Feature":
            selector = classifiers.copy()
            col_selector = "Classifier"
        elif plot_type == "Classifier":
            selector = features.copy()
            col_selector = "Feature"
            hue_order = ["Ensemble", "Conventional", "GCN", "Chance"]
        for selection in tqdm(selector):
            if plot_type == "Classifier":
                selection = fixed_features[selection]
            df_ = df[df[col_selector] == selection]
            df_["Setting"] = np.where(
                df_["Classifier"] == "GCN",
                df_["Setting"].replace("Conventional", "GCN"),
                df_["Setting"],
            )
            fig = sns.relplot(
                data=df_,
                x="n_samples_per_class",
                y=metric,
                col="dataset",
                hue="Setting",
                hue_order=hue_order,
                style=plot_type,
                palette=colors,
                kind="line",
                facet_kws={
                    "sharey": False,
                    "sharex": False,
                },
                col_wrap=3,
                markers=True,
            )
            fig.fig.subplots_adjust(hspace=1)
            for i, ax in enumerate(fig.axes.flatten()):
                title = f"{fixed_datasets[datasets[i]]}, {n_subs[i]} subjects,\n {n_samples[i]} samples"
                ax.text(
                    0.5,
                    1.2,
                    title,
                    transform=ax.transAxes,
                    horizontalalignment="center",
                    verticalalignment="center",
                )
                ax.set_title("")
            fig.set_ylabels("Accuracy (%)", clear_inner=False)
            fig.set_xlabels("No. of samples per class", clear_inner=False)
            sns.move_legend(
                fig,
                "lower center",
                ncol=1,
                frameon=True,
                shadow=True,
                bbox_to_anchor=(0.7, 0.1),
                title=None,
            )
            plt.savefig(
                os.path.join(
                    out_dir,
                    f"samples_per_class_all_datasets_{metric}_{plot_type}_{selection}.png",
                ),
                bbox_inches="tight",
            )
            plt.savefig(
                os.path.join(
                    out_dir,
                    f"samples_per_class_all_datasets_{metric}_{plot_type}_{selection}.svg",
                ),
                bbox_inches="tight",
            )
            plt.close()
