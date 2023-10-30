import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob

sns.set_context("talk", font_scale=1)

# Define variables
# datasets
datasets = ["bold5000", "neuromod", "rsvp", "forrest"]
# classifiers
classifiers = ["LinearSVC", "RandomForest"]
DATA_ROOT = "/storage/store2/work/haggarwa/retreat_2023"
out_dir = os.path.join(DATA_ROOT, "plots")
os.makedirs(out_dir, exist_ok=True)

for dataset in datasets:
    results_dir = os.path.join(DATA_ROOT, f"results_{dataset}")
    for classifier in classifiers:
        pickles = glob(os.path.join(results_dir, f"*{classifier}*.pkl"))
        n_subs = len(pickles)
        dfs = []
        for pickle in pickles:
            df = pd.read_pickle(pickle)
            dfs.append(df)
        n_samples = int(df["train_size"].max() // 0.9)
        df = pd.concat(dfs)
        df = df.reset_index(drop=True)
        sns.pointplot(data=df, x="train_size", y="accuracy", hue="setting")
        plt.axhline(y=df["dummy_accuracy"].mean(), color="k", linestyle="--")
        plt.ylabel("Accuracy")
        plt.xlabel("Training size")
        plt.title(f"{dataset}, {n_subs} subjects, {n_samples} samples")
        plt.savefig(
            os.path.join(out_dir, f"{dataset}_{classifier}.png"),
            bbox_inches="tight",
        )
        plt.close()
