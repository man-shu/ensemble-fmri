"""
This script plots fig4 from the paper. 
It contains the gain in accuracy of the ensemble over the conventional
setting vs. the number of subjects in the ensemble.
"""

import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from glob import glob
import numpy as np
from tqdm import tqdm

sns.set_context("talk", font_scale=1.2)


def fix_names(df):
    if "setting" in list(df.columns):
        df["Setting"] = df["setting"].apply(
            lambda x: "Conventional" if x == "conventional" else "Ensemble"
        )
    elif "Setting" in list(df.columns):
        df["Setting"] = df["Setting"].apply(
            lambda x: "Conventional" if x == "conventional" else "Ensemble"
        )
    if "feature" in list(df.columns):
        df["Feature"] = df["feature"].apply(
            lambda x: "Voxels" if x == "voxels" else "DiFuMo"
        )
    elif "Feature" in list(df.columns):
        df["Feature"] = df["Feature"].apply(
            lambda x: "Voxels" if x == "voxels" else "DiFuMo"
        )
    return df


def calculate_gain(df, dataset, feature, classifier):
    gain_dfs = []
    n_sub_variants = df["n_stacked"].unique()
    for n_sub_variant in n_sub_variants:
        gain = {
            "gain": [],
            "n_stacked": [],
            "n_samples": [],
            "dataset": [],
            "Feature": [],
            "Classifier": [],
        }
        df_ = df[df["n_stacked"] == n_sub_variant]

        acc_conventional = df_[df_["Setting"] == "Conventional"][
            "balanced_accuracy"
        ].values
        acc_stacked = df_[df_["Setting"] == "Ensemble"][
            "balanced_accuracy"
        ].values
        if len(acc_conventional) != len(acc_stacked):
            true_labels_conv = np.concatenate(
                df_[df_["Setting"] == "Conventional"]["true"].to_list()[:50]
            )
            true_labels_ensemble = np.concatenate(
                df_[df_["Setting"] == "Ensemble"]["true"].to_list()
            )
            acc_conventional = acc_conventional[: len(acc_stacked)]
        gains = acc_stacked - acc_conventional
        gain["gain"].extend(gains)
        gain["n_stacked"].extend([n_sub_variant] * len(gains))
        gain["n_samples"].extend(
            df_[df_["Setting"] == "Ensemble"]["train_size"].values
        )
        gain["dataset"].extend([dataset] * len(gains))
        gain["Feature"].extend([feature] * len(gains))
        gain["Classifier"].extend([classifier] * len(gains))
        gain_df = pd.DataFrame(gain)
        gain_dfs.append(gain_df)
    gain_df = pd.concat(gain_dfs)
    return gain_df


def load_all_subs(
    dataset,
    feature,
    classifier,
    results_root,
):
    max_sub_dir = os.path.join(results_root, f"{dataset}_{feature}")
    vary_sub_dir = os.path.join(results_root, f"{dataset}_{feature}_varysubs")
    max_sub_pkls = glob(os.path.join(max_sub_dir, f"*{classifier}*.pkl"))
    vary_sub_pkls = glob(os.path.join(vary_sub_dir, f"*{classifier}*.pkl"))
    subs = np.array(
        [
            os.path.basename(pkl).split("_")[-1].split(".")[0]
            for pkl in max_sub_pkls
        ]
    )
    n_subs = len(subs)
    dfs = []
    gain_dfs = []
    for max_sub, vary_sub in tqdm(
        zip(max_sub_pkls, vary_sub_pkls),
        desc=f"Loading {feature} {classifier} results",
        total=len(max_sub_pkls),
    ):
        max_sub_df = pd.read_pickle(max_sub)
        max_sub_df["n_stacked"] = [n_subs] * max_sub_df.shape[0]
        max_sub_df["subs_stacked"] = [subs] * max_sub_df.shape[0]
        vary_sub_df = pd.read_pickle(vary_sub)
        n_sub_variants = vary_sub_df["n_stacked"].unique()
        conv_dfs = []
        for n_sub_variant in n_sub_variants:
            conventional_df = max_sub_df[
                max_sub_df["setting"] == "conventional"
            ]
            conventional_df.loc[:, "n_stacked"] = [
                n_sub_variant
            ] * conventional_df.shape[0]
            conv_dfs.append(conventional_df)
        df = pd.concat([max_sub_df, vary_sub_df, *conv_dfs])
        df = fix_names(df)
        gain_df = calculate_gain(df, dataset, feature, classifier)
        gain_dfs.append(gain_df)
        dfs.append(df)
    df = pd.concat(dfs)
    df.reset_index(drop=True, inplace=True)

    gain_df = pd.concat(gain_dfs)
    gain_df.reset_index(drop=True, inplace=True)
    return df, gain_df


if __name__ == "__main__":
    DATA_ROOT = "."

    results_root = os.path.join(DATA_ROOT, "results")
    out_dir = os.path.join(DATA_ROOT, "plots")
    os.makedirs(out_dir, exist_ok=True)

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
    n_subs = [4, 203, 10, 3, 13]

    gain_dfs = []
    dfs = []
    for dataset_i, dataset in enumerate(datasets):
        for feature in ["voxels", "difumo"]:
            for classifier in ["MLP", "LinearSVC", "RandomForest"]:
                df, gain_df = load_all_subs(
                    dataset, feature, classifier, results_root
                )
                dfs.append(df)
                gain_dfs.append(gain_df)
    df = pd.concat(dfs)
    df.reset_index(drop=True, inplace=True)

    gain_df = pd.concat(gain_dfs)
    gain_df.reset_index(drop=True, inplace=True)
    n_class = np.unique(np.concatenate(df["true"].to_list())).shape[0]
    # add n_class and n_samples_per_class columns
    df["n_classes"] = [n_class] * df.shape[0]
    df["n_samples_per_class"] = df["train_size"] / df["n_classes"]
    gain_df["n_classes"] = [n_class] * gain_df.shape[0]
    gain_df["n_samples_per_class"] = (
        gain_df["n_samples"] / gain_df["n_classes"]
    )
    gain_df["gain"] = gain_df["gain"] * 100
    gain_df = fix_names(gain_df)

    colors = sns.color_palette("tab10")
    mlp_colors = colors[2]
    lsvc_colors = colors[3]
    rf_colors = colors[4]
    colors = [mlp_colors, lsvc_colors, rf_colors]
    fig = sns.relplot(
        data=gain_df,
        x="n_stacked",
        y="gain",
        style="Feature",
        col="dataset",
        hue="Classifier",
        kind="line",
        facet_kws={
            "sharey": False,
            "sharex": False,
        },
        palette=colors,
        # err_style="bars",
        markers=True,
        col_wrap=3,
    )
    for i, ax in enumerate(fig.axes.ravel()):
        title = f"{fixed_datasets[datasets[i]]}, {n_subs[i]} subjects,\n {n_samples[i]} samples"
        ax.set_title(
            f"{fixed_datasets[datasets[i]]}, {n_subs[i]} subjects,\n {n_samples[i]} samples"
        )
        if datasets[i] == "aomic_anticipation":
            ax.set(xscale="log")
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
            ax.ticklabel_format(style="plain")
        else:
            ax.set(xscale="linear")
            if n_subs[i] > 5:
                ax.get_xaxis().set_major_locator(MultipleLocator(2))
            else:
                ax.get_xaxis().set_major_locator(MultipleLocator(1))
        ax.set_xlim(
            1,
        )
        ax.axhline(0, color="k")
        ax.axvline(10, color="k")
    fig.fig.subplots_adjust(hspace=0.5)
    fig.set_xlabels("No. of subjects")
    fig.set_ylabels("Gain (%)")
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
        os.path.join(out_dir, "fig4.png"),
        bbox_inches="tight",
    )
    plt.savefig(
        os.path.join(out_dir, "fig4.svg"),
        bbox_inches="tight",
    )
    plt.close()
