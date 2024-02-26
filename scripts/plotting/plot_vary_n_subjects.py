import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
from tqdm import tqdm
from sklearn.utils import Bunch
import pdb
from scipy.optimize import curve_fit

sns.set_context("talk", font_scale=1.2)
DATA_ROOT = "/storage/store2/work/haggarwa/retreat_2023"


# Define the logistic function
def logistic(x, a, b, c):
    return c / (1 + np.exp(-(x - b) / a))


def get_logistic_fit(gain_df, classifier):
    mask = gain_df["Classifier"] == classifier
    x = gain_df.loc[mask, "n_stacked"].values
    y = gain_df.loc[mask, "gain"].values
    popt, pcov = curve_fit(logistic, x, y)
    gain_df.loc[mask, "logistic_fit"] = logistic(
        gain_df["n_stacked"].values, *popt
    )

    return gain_df


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
            if not np.array_equal(true_labels_conv, true_labels_ensemble):
                pdb.set_trace()
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
        try:
            gain_df = pd.DataFrame(gain)
        except Exception as error:
            print(error)
            pdb.set_trace()
        gain_dfs.append(gain_df)
    gain_df = pd.concat(gain_dfs)
    return gain_df


def load_all_subs(
    dataset,
    feature,
    classifier,
    results_root=os.path.join(DATA_ROOT, "results"),
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
    results_root = os.path.join(DATA_ROOT, "results")
    out_dir = os.path.join(DATA_ROOT, "plots")
    os.makedirs(out_dir, exist_ok=True)

    datasets = ["aomic_anticipation"]

    # Camelized dataset names
    fixed_datasets = {
        "neuromod": "Neuromod",
        "forrest": "Forrest",
        "bold5000": "BOLD5000",
        "rsvp": "RSVP-IBC",
        # "nsd": "NSD",
        "aomic_anticipation": "AOMIC",
    }
    n_samples = [61]

    gain_dfs = []
    dfs = []
    for dataset_i, dataset in enumerate(datasets):
        for feature in ["voxels", "difumo"]:
            for classifier in ["LinearSVC", "RandomForest"]:
                df, gain_df = load_all_subs(dataset, feature, classifier)
                gain_df = get_logistic_fit(gain_df, classifier)
                dfs.append(df)
                gain_dfs.append(gain_df)
    df = pd.concat(dfs)
    df.reset_index(drop=True, inplace=True)

    n_subs = df["subject"].unique().shape[0]

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
    gain_df["logistic_fit"] = gain_df["logistic_fit"] * 100
    # sns.relplot(
    #     data=df,
    #     x="n_samples_per_class",
    #     y="balanced_accuracy",
    #     col="n_stacked",
    #     hue="setting",
    #     style="classifier",
    #     palette=["b", "r"],
    #     kind="line",
    # )
    # plt.savefig(
    #     os.path.join(out_dir, f"varysubs_vs_accuracy_{feature}.png")
    # )
    # plt.savefig(
    #     os.path.join(out_dir, f"varysubs_vs_accuracy_{feature}.svg")
    # )
    # plt.close()

    # fig = sns.lmplot(
    #     data=gain_df,
    #     x="n_stacked",
    #     y="gain",
    #     col="dataset",
    #     hue="Classifier",
    #     logistic=True,
    #     x_estimator=np.mean,
    #     aspect=1.5,
    #     facet_kws={
    #         "sharey": False,
    #         "sharex": False,
    #     },
    # )
    gain_df = fix_names(gain_df)
    fig = sns.relplot(
        data=gain_df,
        x="n_stacked",
        y="gain",
        style="Classifier",
        col="dataset",
        hue="Feature",
        kind="line",
        facet_kws={
            "sharey": False,
            "sharex": False,
        },
        aspect=1.5,
        palette=["b", "r"],
        # err_style="bars",
        markers=True,
    )
    for i, ax in enumerate(fig.axes.ravel()):
        title = f"{fixed_datasets[dataset]}, {n_subs} subjects,\n {n_samples[dataset_i]} samples"
        ax.set_title(
            f"{fixed_datasets[dataset]}, {n_subs} subjects,\n {n_samples[dataset_i]} samples"
        )
        ax.set(xscale="log")
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.ticklabel_format(style="plain")
        ax.set_xlabel("No. of subjects in the ensemble")
        ax.set_ylabel("Gain (%)")
        ax.axhline(0, color="k")
        ax.axvline(10, color="k")
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
        os.path.join(out_dir, "varysubs_vs_gain.png"),
        bbox_inches="tight",
    )
    plt.savefig(
        os.path.join(out_dir, "varysubs_vs_gain.svg"),
        bbox_inches="tight",
    )
    plt.close()
