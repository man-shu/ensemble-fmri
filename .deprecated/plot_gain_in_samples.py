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

DATA_ROOT = "."
out_dir = os.path.join(DATA_ROOT, "plots_copy")
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

                    dfs.append(df)
                    dfs.append(pd.DataFrame(chance_rows))

        n_samples.append(n_sample)
        n_subs.append(n_sub)
        n_classes.append(n_class)

    ### start plotting
    df = pd.concat(dfs)
    df = df.reset_index(drop=True)

    datasets = [
        "neuromod",
        "aomic_anticipation",
        "forrest",
        "bold",
        "rsvp",
    ]

    for dataset in datasets:
        print(dataset)
        df_neuromod = df[df["dataset"] == dataset]
        df_neuromod = df_neuromod.reset_index(drop=True)

        df_neuromod_ensemble = df_neuromod[
            df_neuromod["Setting"] == "Ensemble"
        ]
        df_neuromod_ensemble = df_neuromod_ensemble.groupby(
            ["train_size"]
        ).mean()
        df_neuromod_ensemble = df_neuromod_ensemble.reset_index(drop=True)
        df_neuromod_ensemble[metric] = df_neuromod_ensemble[metric].round(1)

        # print(df_neuromod_ensemble)

        df_neuromod_conventional = df_neuromod[
            df_neuromod["Setting"] == "Conventional"
        ]
        df_neuromod_conventional = df_neuromod_conventional.groupby(
            ["train_size"]
        ).mean()
        df_neuromod_conventional = df_neuromod_conventional.reset_index(
            drop=True
        )
        df_neuromod_conventional[metric] = df_neuromod_conventional[
            metric
        ].round(1)

        # print(df_neuromod_conventional)

        acc_equivalent = {
            "acc_ensemble": [],
            "acc_conventional": [],
            "train_size_ensemble": [],
            "train_size_conventional": [],
        }
        for i, row_ens in df_neuromod_ensemble.iterrows():
            for j, row_con in df_neuromod_conventional.iterrows():
                if row_ens[metric] == row_con[metric]:
                    acc_equivalent["acc_ensemble"].append(row_ens[metric])
                    acc_equivalent["acc_conventional"].append(row_con[metric])
                    acc_equivalent["train_size_ensemble"].append(
                        row_ens["n_samples_per_class"]
                    )
                    acc_equivalent["train_size_conventional"].append(
                        row_con["n_samples_per_class"]
                    )
        # print(acc_equivalent)

        diff = np.array(acc_equivalent["train_size_conventional"]) - np.array(
            acc_equivalent["train_size_ensemble"]
        )

        print("\n\n\n*****")
        print(
            dataset,
            np.mean(diff),
        )
        print("*****\n\n\n")
