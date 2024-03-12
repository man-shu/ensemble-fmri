import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
from sklearn.utils import Bunch
from tqdm import tqdm
from scipy.optimize import curve_fit
import pdb
from seaborn._stats.base import Stat
from seaborn._stats.aggregation import Est
from matplotlib.ticker import MultipleLocator


# Define the log function
def log_func(x, a, b):
    return a * np.log(x) + b


def get_lm_fit(gain_df, classifier, plot_type):
    mask = gain_df[plot_type] == classifier
    gain_df.loc[mask, "lm_fit"] = np.nan
    samples = gain_df["n_samples_per_class"].unique()
    y = []
    for sample in samples:
        mask_sample = gain_df.loc[mask, "n_samples_per_class"] == sample
        y_sample = np.mean(gain_df.loc[mask].loc[mask_sample, "gain"].values)
        y.append(y_sample)
    popt, pcov = curve_fit(log_func, samples, y)
    y_fit = log_func(samples, *popt)
    for i, sample in enumerate(samples):
        for j, row in gain_df.iterrows():
            if row["n_samples_per_class"] == sample:
                row["lm_fit"] = y_fit[i]
    pdb.set_trace()
    return gain_df


sns.set_context("talk", font_scale=1.2)
# datasets
datasets = [
    "neuromod",
    "aomic_anticipation",
    "forrest",
    "bold5000",
    "rsvp",
    # "nsd",
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
n_samples = [50, 61, 175, 332, 360]
# performance metrics
# metrics = ["accuracy", "balanced_accuracy"]
metrics = ["balanced_accuracy"]
# features
features = ["voxels", "difumo"]
# fixed feature names
fixed_features = {"voxels": "Voxels", "difumo": "DiFuMo"}
# classifiers
classifiers = ["MLP", "LinearSVC", "RandomForest"]
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

DATA_ROOT = "/storage/store2/work/haggarwa/retreat_2023"
out_dir = os.path.join(DATA_ROOT, "plots_l2_penalty_in_pretraining")
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

    for order, (dataset, n_sample) in tqdm(
        enumerate(zip(datasets, n_samples)), total=len(datasets)
    ):
        for feature in features:
            results_dir = os.path.join(
                DATA_ROOT, "results_l2_penalty_in_pretraining", f"{dataset}_{feature}"
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
    gains["n_samples_per_class"] = gains["n_samples"] / gains["n_classes"]

    # fix names
    gains["Feature"] = gains["Feature"].map(fixed_features)
    gains["gain"] = gains["gain"] * 100

    av_gains = (
        gains.groupby(["dataset", "Feature", "Classifier"])
        .mean()
        .reset_index()
    )
    print(av_gains)

    ### start plotting
    df = pd.concat(dfs)
    df = df.reset_index(drop=True)
    df[metric] = df[metric] * 100

    ### save average accuracy tables
    av_acc = (
        df.groupby(["dataset", "Setting", "Feature", "Classifier"])
        .mean()
        .reset_index()
    )
    # get std deviation
    std_acc = (
        df.groupby(["dataset", "Setting", "Feature", "Classifier"])
        .std()
        .reset_index()
    )
    for dataframe, typ in zip([av_acc, std_acc], ["mean", "std"]):
        dataframe["Dataset"] = dataframe["dataset"].map(fixed_datasets)
        dataframe = dataframe[
            ["Dataset", "Feature", "Classifier", "Setting", metric, "Order"]
        ]
        dataframe = dataframe.round(1)
        dataframe = dataframe.pivot(
            index=["Dataset", "Order", "Feature", "Classifier"],
            columns="Setting",
            values=metric,
        )
        dataframe = dataframe.sort_values(by="Order")
        dataframe.to_csv(
            os.path.join(out_dir, f"{typ}_{metric}.csv"),
            index=True,
            header=True,
        )

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
    colors = sns.color_palette("tab20c")
    ensemble_colors = colors[:3]
    conventional_colors = colors[4:7]
    colors = ensemble_colors + conventional_colors

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
                # weight="bold",
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
        bbox_to_anchor=(2.3, 0),
        title="Setting, Classifier",
        fontsize="small",
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

    ### lineplots over varying training size
    colors = sns.color_palette("tab10")
    ensemble_colors = colors[0]
    conventional_colors = colors[1]
    colors = [ensemble_colors, conventional_colors, "k"]
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
                x="n_samples_per_class",
                y=metric,
                col="dataset",
                hue="Setting",
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
                # ax.set(
                #     xscale="log",
                #     # yscale="log",
                #     # ylim=(0, 1),
                #     # xlim=(0, 5000),
                # )
                # ax.set_xscale("log", base=2)
                # if n_samples[i] > 100:
                #     ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
                #     ax.ticklabel_format(style="plain")
                #     fmt = ax.get_xaxis().get_major_formatter()
                # else:
                #     ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
                #     fmt = ax.get_xaxis().get_major_formatter()
                #     fmt.set_scientific(False)
                #     print(fmt.__dict__)

                title = f"{fixed_datasets[datasets[i]]}, {n_subs[i]} subjects,\n {n_samples[i]} samples"
                ax.text(
                    0.5,
                    1.7,
                    title,
                    transform=ax.transAxes,
                    horizontalalignment="center",
                    verticalalignment="center",
                )
                stats = "Average gain:\n"
                values = av_gains_[av_gains_["dataset"] == datasets[i]][
                    "gain"
                ].values.astype(int)
                keys = av_gains_[av_gains_["dataset"] == datasets[i]][
                    plot_type
                ].values

                ax.set_title("")
                for i, (key, value) in enumerate(zip(keys, values)):
                    stats += f"{key}: {value}%\n"
                ax.set_title(stats, loc="right", fontsize="small")
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

    ### scatter plots of gains vs. n_samples_per_class
    colors = sns.color_palette("tab10")
    mlp_colors = colors[2]
    lsvc_colors = colors[3]
    rf_colors = colors[4]
    colors = [mlp_colors, lsvc_colors, rf_colors]
    fig = sns.relplot(
        data=gains,
        x="n_samples_per_class",
        y="gain",
        col="dataset",
        hue="Classifier",
        # size="n_subs",
        style="Feature",
        kind="line",
        # col_wrap=2,
        palette=colors,
        facet_kws={
            "sharey": False,
            "sharex": False,
        },
        # err_style="bars",
        markers=True,
        col_wrap=3,
    )
    for i, ax in enumerate(fig.axes.flatten()):
        ax.set_xlim(
            1,
        )
        title = f"{fixed_datasets[datasets[i]]}, {n_subs[i]} subjects,\n {n_samples[i]} samples"
        ax.set_title(title)
        ax.axhline(0, color="k", label="No gain")
        ax.axvline(10, color="k", label="10 sample per class")
        # ax.set(xscale="log")
        # ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        # ax.ticklabel_format(style="plain")

        samples_per_class = n_samples[i] / n_classes[i]

        if samples_per_class > 40:
            ax.get_xaxis().set_major_locator(MultipleLocator(20))
        elif samples_per_class > 20 and samples_per_class < 40:
            ax.get_xaxis().set_major_locator(MultipleLocator(10))
        else:
            ax.get_xaxis().set_major_locator(MultipleLocator(2))

    fig.set_ylabels("Gain (%)", clear_inner=False)
    fig.set_xlabels("No. of samples per class", clear_inner=False)
    fig.fig.subplots_adjust(hspace=0.5)
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
            f"gains_v_samples_per_class_{metric}.png",
        ),
        bbox_inches="tight",
    )
    plt.savefig(
        os.path.join(
            out_dir,
            f"gains_v_samples_per_class_{metric}.svg",
        ),
        bbox_inches="tight",
    )
    plt.close()

    ### lmplots of gains vs. n_samples_per_class

    # types = gains_[plot_type].unique()
    # for t in types:
    #     gains_ = get_lm_fit(gains_, t, plot_type)

    # fig = sns.lmplot(
    #     data=gains,
    #     x="n_samples_per_class",
    #     y="gain",
    #     col="dataset",
    #     # style=plot_type,
    #     hue=plot_type,
    #     aspect=1.5,
    #     logx=True,
    #     x_estimator=np.mean,
    #     facet_kws={
    #         "sharey": False,
    #         "sharex": False,
    #     },
    #     # kind="line",
    #     palette=["g", "g"],
    #     # err_style="bars",
    #     markers=["o", "x"],
    #     scatter_kws=dict(s=100),
    #     # dashes=False,
    #     # linewidth=0,
    # )

    # for i, ax in enumerate(fig.axes.flatten()):
    #     ax.set_xlim(
    #         1,
    #     )
    #     title = f"{fixed_datasets[datasets[i]]}, {n_subs[i]} subjects,\n {n_samples[i]} samples"
    #     ax.set_title(title)
    #     ax.lines[-1].set(linestyle="--")
    #     ax.axhline(0, color="k", label="No gain")
    #     ax.axvline(10, color="k", label="10 sample per class")
    #     # pdb.set_trace()
    #     # ax.set(xscale="log")

    # fig.set_ylabels("Gain (%)", clear_inner=False)
    # fig.set_xlabels("Number of samples per class", clear_inner=False)

    # # legend = fig.legend
    # # for i, handle in enumerate(legend.legend_handles):
    # #     if i == 1:
    # #         handle.set_linestyle("--")

    # plt.savefig(
    #     os.path.join(
    #         out_dir,
    #         f"gains_v_samples_per_class_lm_{metric}_{plot_type}_{selection}.png",
    #     ),
    #     bbox_inches="tight",
    # )
    # plt.savefig(
    #     os.path.join(
    #         out_dir,
    #         f"gains_v_samples_per_class_lm_{metric}_{plot_type}_{selection}.svg",
    #     ),
    #     bbox_inches="tight",
    # )
    # plt.close()
