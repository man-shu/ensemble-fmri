import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import time
from glob import glob
import importlib.util
import sys
from sklearn.utils import Bunch
from utils import parcellate, pretrain, decode, generate_sub_clf_combinations

DATA_ROOT = "/storage/store2/work/haggarwa/retreat_2023/data/"
OUT_ROOT = "/storage/store2/work/haggarwa/retreat_2023"

# load local utility functions
# spec = importlib.util.spec_from_file_location(
#     "utils",
#     os.path.join(OUT_ROOT, "utils.py"),
# )
# utils = importlib.util.module_from_spec(spec)
# sys.modules["utils"] = utils
# spec.loader.exec_module(utils)

# datasets and classifiers to use
# datas = [
#     "bold5000",
#     "bold5000_fold2",
#     "bold5000_fold3",
#     "bold5000_fold4",
#     "forrest",
#     "neuromod",
#     "rsvp_trial",
# ]
datas = [
    "bold5000",
    # "bold5000_fold2",
    # "bold5000_fold3",
    # "bold5000_fold4",
    "forrest",
    # "neuromod",
    "rsvp",
]
classifiers = ["LinearSVC", "RandomForest"]

for dataset in datas:
    # input data root path
    data_dir = os.path.join(DATA_ROOT, dataset)
    data_resolution = "3mm"  # or 1_5mm
    nifti_dir = os.path.join(data_dir, data_resolution)

    # create fake, empty atlas object
    atlas = Bunch()
    atlas.name = "wholebrain"

    # output results path
    start_time = time.strftime("%Y%m%d-%H%M%S")
    results_dir = f"{dataset}_{atlas.name}_results_{start_time}"
    results_dir = os.path.join(OUT_ROOT, results_dir)
    os.makedirs(results_dir, exist_ok=True)

    # get file names
    imgs = glob(os.path.join(nifti_dir, "*.nii.gz"))
    subjects = [os.path.basename(img).split(".")[0] for img in imgs]

    print(f"\nParcellating {dataset}...")
    data = Parallel(n_jobs=len(subjects), verbose=2, backend="loky")(
        delayed(parcellate)(
            imgs[i],
            subject,
            atlas,
            DATA_ROOT=DATA_ROOT,
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
        n_jobs=len(subjects), verbose=2, backend="loky"
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
        n_jobs=len(subjects), verbose=11, backend="loky"
    )(
        delayed(pretrain)(
            subject,
            data,
            dummy=False,
            data_dir=data_dir,
            atlas=atlas,
        )
        for subject in subjects
    )

    print(f"\nRunning cross-val on {dataset}...")
    all_results = Parallel(
        n_jobs=len(subjects) * len(classifiers), verbose=2, backend="loky"
    )(
        delayed(decode)(
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
    # all_results = []
    # for subject, subject_i, clf in generate_sub_clf_combinations(
    #     subjects, classifiers
    # ):
    #     print(subject, subject_i)
    #     result = decode(
    #         subject,
    #         subject_i,
    #         data,
    #         clf,
    #         fitted_classifiers,
    #         dummy_fitted_classifiers,
    #         results_dir,
    #     )
    #     all_results.append(result)

    print(f"\nPlotting results for {dataset}...")
    df = pd.concat(all_results)
    df["setting_classifier"] = df["setting"] + "_" + df["classifier"]

    sns.pointplot(
        data=df, x="train_size", y="accuracy", hue="setting_classifier"
    )
    plt.axhline(y=df["dummy_accuracy"].mean(), color="k", linestyle="--")
    plt.ylabel("Accuracy")
    plt.xlabel("Training size")
    plt.savefig(os.path.join(results_dir, f"accuracy_{start_time}.png"))
    plt.close()

    sns.boxplot(
        data=df, x="train_size", y="accuracy", hue="setting_classifier"
    )
    plt.axhline(y=df["dummy_accuracy"].mean(), color="k", linestyle="--")
    plt.ylabel("Accuracy")
    plt.xlabel("Training size")
    plt.savefig(os.path.join(results_dir, f"box_accuracy_{start_time}.png"))
    plt.close()

    sns.pointplot(
        data=df,
        x="train_size",
        y="balanced_accuracy",
        hue="setting_classifier",
    )
    plt.axhline(
        y=df["dummy_balanced_accuracy"].mean(), color="k", linestyle="--"
    )
    plt.ylabel("Balanced Accuracy")
    plt.xlabel("Training size")
    plt.savefig(
        os.path.join(results_dir, f"balanced_accuracy_{start_time}.png")
    )
    plt.close()

    sns.boxplot(
        data=df,
        x="train_size",
        y="balanced_accuracy",
        hue="setting_classifier",
    )
    plt.axhline(
        y=df["dummy_balanced_accuracy"].mean(), color="k", linestyle="--"
    )
    plt.ylabel("Balanced Accuracy")
    plt.xlabel("Training size")
    plt.savefig(
        os.path.join(results_dir, f"box_balanced_accuracy_{start_time}.png")
    )
    plt.close()
