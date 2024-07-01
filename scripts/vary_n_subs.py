"""
This script performs decoding on varying training sizes and varying number
subjects in the ensemble, for five different datasets: neuromod, forrest, rsvp, 
bold and aomic_anticipation, using either DiFuMo or full-voxel features.

"""

import pandas as pd
from nilearn import datasets
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from glob import glob
import sys

# add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import (
    parcellate,
    pretrain,
    generate_sub_clf_combinations,
    vary_stacked_subs,
)

if len(sys.argv) != 5:
    raise ValueError(
        "Please provide the following arguments in that order: ",
        "path to data, path to output, N parallel jobs, features.\n",
        "For example: ",
        "python scripts/vary_n_subs.py data results 20 voxels\n",
    )
else:
    DATA_ROOT = sys.argv[1]
    OUT_ROOT = sys.argv[2]
    N_JOBS = sys.argv[3]
    features = sys.argv[4]

# datasets and classifiers to use
datas = [
    "neuromod",
    "forrest",
    "rsvp",
    "aomic_anticipation",
    "bold",
]
classifiers = ["LinearSVC", "RandomForest", "MLP"]

for dataset in datas:
    # input data root path
    data_dir = os.path.join(DATA_ROOT, dataset)
    data_resolution = "3mm"  # or 1_5mm
    nifti_dir = os.path.join(data_dir, data_resolution)

    if features == "voxels":
        # create fake, empty atlas object
        atlas = {}
        atlas["name"] = "wholebrain"
    elif features == "difumo":
        # get difumo atlas
        atlas = datasets.fetch_atlas_difumo(
            dimension=1024,
            resolution_mm=3,
            data_dir=DATA_ROOT,
            legacy_format=False,
        )
        atlas["name"] = "difumo"
    else:
        raise ValueError(f"Unknown feature set: {features}")

    # output results path
    results_dir = f"{dataset}_{features}_varysubs"
    results_dir = os.path.join(OUT_ROOT, results_dir)
    os.makedirs(results_dir, exist_ok=True)

    # get file names
    imgs = glob(os.path.join(nifti_dir, "*.nii.gz"))
    subjects = [os.path.basename(img).split(".")[0] for img in imgs]

    print(f"\nParcellating {dataset}...")
    data = Parallel(n_jobs=N_JOBS, verbose=11, backend="multiprocessing")(
        delayed(parcellate)(
            imgs[i],
            subject,
            atlas,
            data_dir=data_dir,
            nifti_dir=nifti_dir,
        )
        for i, subject in enumerate(subjects)
    )

    print(f"\nConcatenating all data for {dataset}...")
    data_ = dict(responses=[], conditions=[], runs=[], subjects=[])
    for entry in data:
        for key in ["responses", "conditions", "runs", "subjects"]:
            data_[key].extend(entry[key])
    for key in ["responses", "conditions", "runs", "subjects"]:
        data_[key] = np.array(data_[key])
    data = data_.copy()
    del data_

    print(f"\nPretraining dummy classifiers on {dataset}...")
    dummy_fitted_classifiers = Parallel(
        n_jobs=N_JOBS, verbose=11, backend="multiprocessing"
    )(
        delayed(pretrain)(
            subject=subject,
            data=data,
            dummy=True,
            data_dir=data_dir,
            atlas=atlas,
        )
        for subject in subjects
    )

    print(f"\nPretraining linear classifiers on {dataset}...")
    fitted_classifiers = Parallel(
        n_jobs=N_JOBS, verbose=11, backend="multiprocessing"
    )(
        delayed(pretrain)(
            subject=subject,
            data=data,
            dummy=False,
            data_dir=data_dir,
            atlas=atlas,
        )
        for subject in subjects
    )

    print(f"\nRunning cross-val on {dataset}...")
    all_results = Parallel(
        n_jobs=N_JOBS,
        verbose=2,
        backend="multiprocessing",
    )(
        delayed(vary_stacked_subs)(
            subject,
            subject_i,
            data,
            clf,
            fitted_classifiers,
            dummy_fitted_classifiers,
            results_dir,
            dataset,
        )
        for subject, subject_i, clf in generate_sub_clf_combinations(
            subjects, classifiers
        )
    )

    print(f"\nPlotting results for {dataset}...")
    df = pd.concat(all_results)
    df["setting_classifier"] = df["setting"] + "_" + df["classifier"]
    sns.pointplot(
        data=df, x="n_stacked", y="accuracy", hue="setting_classifier"
    )
    plt.axhline(y=df["dummy_accuracy"].mean(), color="k", linestyle="--")
    plt.ylabel("Accuracy")
    plt.xlabel("No. of subjects stacked")
    plt.savefig(os.path.join(results_dir, f"accuracy.png"))
    plt.close()

    sns.boxplot(data=df, x="train_size", y="accuracy", hue="n_stacked")
    plt.axhline(y=df["dummy_accuracy"].mean(), color="k", linestyle="--")
    plt.ylabel("Accuracy")
    plt.xlabel("No. of subjects stacked")
    plt.savefig(os.path.join(results_dir, f"box_accuracy.png"))
    plt.close()

    sns.pointplot(
        data=df,
        x="n_stacked",
        y="balanced_accuracy",
        hue="setting_classifier",
    )
    plt.axhline(
        y=df["dummy_balanced_accuracy"].mean(), color="k", linestyle="--"
    )
    plt.ylabel("Balanced Accuracy")
    plt.xlabel("No. of subjects stacked")
    plt.savefig(os.path.join(results_dir, f"balanced_accuracy.png"))
    plt.close()

    sns.pointplot(
        data=df,
        x="n_stacked",
        y="balanced_accuracy",
        hue="setting_classifier",
    )
    plt.axhline(
        y=df["dummy_balanced_accuracy"].mean(), color="k", linestyle="--"
    )
    plt.ylabel("Balanced Accuracy")
    plt.xlabel("No. of subjects stacked")
    plt.savefig(os.path.join(results_dir, f"balanced_accuracy.png"))
    plt.close()

    sns.boxplot(
        data=df,
        x="n_stacked",
        y="balanced_accuracy",
        hue="setting_classifier",
    )
    plt.axhline(
        y=df["dummy_balanced_accuracy"].mean(), color="k", linestyle="--"
    )
    plt.ylabel("Balanced Accuracy")
    plt.xlabel("No. of subjects stacked")
    plt.savefig(os.path.join(results_dir, f"box_balanced_accuracy.png"))
    plt.close()
