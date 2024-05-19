import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from joblib import load
from glob import glob

out_dir = (
    "/storage/store2/work/haggarwa/retreat_2023/gcn/grid_search_analysis/"
)
os.makedirs(out_dir, exist_ok=True)

df = pd.read_pickle(
    "/storage/store2/work/haggarwa/retreat_2023/gcn/results/neuromod_difumo_results_20240518-222444/param_sweep/training_metrics_sub-01.pkl"
)
df.to_dict()

print(df.keys())

index = 0
var1 = "batch_size"
var2 = "lr"
var3 = "weight_decay"

combi_keys = {}
combi_cnt = 0
for i in range(len(df)):
    if (i == 0) or (i % 5 == 0):
        aggregate = {
            "train_accuracy": [],
            "valid_accuracy": [],
            "train_loss": [],
            "valid_loss": [],
        }
        combi_cnt += 1
    for key in aggregate.keys():
        aggregate[key].append(df[key][i])
    combi_keys[combi_cnt] = df["params"][i].copy()
    if ((i + 1) % 5) == 0:
        max_train_acc = np.max(
            np.mean(np.array(aggregate["train_accuracy"]), axis=0)
        )
        max_valid_acc = np.max(
            np.mean(np.array(aggregate["valid_accuracy"]), axis=0)
        )
        ninety_train_acc = np.percentile(
            np.mean(np.array(aggregate["train_accuracy"]), axis=0), 90
        )
        ninety_valid_acc = np.percentile(
            np.mean(np.array(aggregate["valid_accuracy"]), axis=0), 90
        )
        plt.plot(
            np.mean(np.array(aggregate["train_accuracy"]), axis=0),
            label="train accuracy",
        )
        plt.plot(
            np.mean(np.array(aggregate["valid_accuracy"]), axis=0),
            label="valid_accuracy",
        )
        params = df["params"][i].copy()
        plt.suptitle(
            f"{var1}: {params[var1]}, {var2}: {params[var2]},\n{var3}: {params[var3]},\n top ninety train acc: {ninety_train_acc:.2f}, top ninety valid acc: {ninety_valid_acc:.2f}",
            fontsize=10,
        )
        plt.legend()
        plt.savefig(os.path.join(out_dir, f"accuracy_{combi_cnt}.png"))
        plt.close()

        min_train_loss = np.min(
            np.mean(np.array(aggregate["train_loss"]), axis=0)
        )
        min_valid_loss = np.min(
            np.mean(np.array(aggregate["valid_loss"]), axis=0)
        )
        ten_train_loss = np.percentile(
            np.mean(np.array(aggregate["train_loss"]), axis=0), 10
        )
        ten_valid_loss = np.percentile(
            np.mean(np.array(aggregate["valid_loss"]), axis=0), 10
        )
        plt.plot(
            np.mean(np.array(aggregate["train_loss"]), axis=0),
            label="train loss",
        )
        plt.plot(
            np.mean(np.array(aggregate["valid_loss"]), axis=0),
            label="valid loss",
        )
        params = df["params"][i].copy()
        plt.suptitle(
            f"{var1}: {params[var1]}, {var2}: {params[var2]},\n{var3}: {params[var3]}, \n least ten perc train loss: {ten_train_loss:.2f}, least ten perc valid loss: {ten_valid_loss:.2f}, \n ",
            fontsize=10,
        )
        plt.legend()
        plt.savefig(os.path.join(out_dir, f"loss_{combi_cnt}.png"))
        plt.close()


# print out the best combinations
sweep_dir = "/storage/store2/work/haggarwa/retreat_2023/gcn/results/neuromod_difumo_results_20240519-075217/param_sweep"
sweep_files = glob(os.path.join(sweep_dir, "best*.pkl"))
keys_to_print = ["batch_size", "lr", "weight_decay", "epochs"]
for f in sweep_files:
    params = load(f)
    print(f"\n\n{f}\n")
    for key in keys_to_print:
        print(f"\t{key}: {params[key]}\n")
