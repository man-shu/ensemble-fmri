import pandas as pd
from nilearn import datasets, maskers
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import time
from glob import glob
import importlib.util
import sys
from tqdm import tqdm
from sklearn.model_selection import (
    StratifiedShuffleSplit,
)
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score


def cross_val(
    subject,
    data,
    classifier,
    results_dir,
    dataset,
):
    results = []
    # select data for current subject
    sub_mask = np.where(data["subjects"] == subject)[0]
    X = data["responses"][sub_mask]
    Y = data["conditions"][sub_mask]
    groups = data["runs"][sub_mask]
    cv = StratifiedShuffleSplit(test_size=0.20, random_state=0, n_splits=5)
    # create conventional classifier
    if classifier == "LinearSVC":
        clf = LinearSVC(dual="auto")
    elif classifier == "RandomForest":
        clf = RandomForestClassifier(n_estimators=200, random_state=0)
    dummy_clf = DummyClassifier(strategy="most_frequent")
    utils._plot_cv_indices(
        cv,
        X,
        Y,
        groups,
        subject,
        5,
        results_dir,
    )
    count = 0
    for train, test in tqdm(
        cv.split(X, Y, groups=groups),
        desc=f"{dataset}, {subject}, {classifier}",
        position=0,
        leave=True,
        total=cv.get_n_splits(),
    ):
        conventional_result = utils._classify(
            clf,
            dummy_clf,
            train,
            test,
            X,
            Y,
            "conventional",
            20,
            classifier,
            subject,
        )
        conventional_result["balanced_accuracy"] = balanced_accuracy_score(
            conventional_result["true"], conventional_result["predicted"]
        )
        conventional_result[
            "balanced_dummy_accuracy"
        ] = balanced_accuracy_score(
            conventional_result["true"], conventional_result["dummy_predicted"]
        )
        results.append(conventional_result)
        print(
            f"{classifier} {20}% left-out, {subject}, split {count} :",
            f"{conventional_result['balanced_accuracy']:.2f} / {conventional_result['balanced_dummy_accuracy']:.2f}",
        )

        count += 1

    results = pd.DataFrame(results)
    results.to_pickle(
        os.path.join(results_dir, f"results_clf_{classifier}_{subject}.pkl")
    )
    return results


if __name__ == "__main__":
    DATA_ROOT = "/storage/store2/work/haggarwa/retreat_2023/data/"
    OUT_ROOT = "/storage/store2/work/haggarwa/retreat_2023"

    # load local utility functions
    spec = importlib.util.spec_from_file_location(
        "utils",
        os.path.join(OUT_ROOT, "utils.py"),
    )
    utils = importlib.util.module_from_spec(spec)
    sys.modules["utils"] = utils
    spec.loader.exec_module(utils)

    dataset = "nsd"
    classifier = "LinearSVC"

    data_dir = os.path.join(DATA_ROOT, dataset)
    data_resolution = "3mm"  # or 1_5mm
    nifti_dir = os.path.join(data_dir, data_resolution)

    # output results path
    start_time = time.strftime("%Y%m%d-%H%M%S")
    results_dir = f"{dataset}_test_results_{start_time}"
    results_dir = os.path.join(OUT_ROOT, results_dir)
    os.makedirs(results_dir, exist_ok=True)

    # get file names
    imgs = glob(os.path.join(nifti_dir, "*.nii.gz"))
    subjects = [os.path.basename(img).split(".")[0] for img in imgs]

    datas = []
    for img in tqdm(imgs):
        masker = maskers.NiftiMasker(n_jobs=20)
        data = masker.fit_transform(img)
        datas.append(data)

    conditions = []
    runs = []
    subs = []
    for i, data in enumerate(datas):
        condition, run, sub = utils._get_labels(subjects[i], data, nifti_dir)
        conditions.append(condition)
        runs.append(run)
        subs.append(sub)

    data = dict(responses=[], conditions=[], runs=[], subjects=[])
    data["responses"] = np.concatenate(datas)
    del datas
    data["conditions"] = np.concatenate(conditions)
    data["runs"] = np.concatenate(runs)
    data["subjects"] = np.concatenate(subs)

    print(f"\nRunning cross-val on {dataset}...")
    # all_results = Parallel(n_jobs=len(subjects), verbose=11, backend="loky")(
    #     delayed(cross_val)(
    #         subject,
    #         data,
    #         classifier,
    #         results_dir,
    #         dataset,
    #     )
    #     for subject in subjects
    # )
    all_results = []
    for subject in subjects:
        results = cross_val(
            subject,
            data,
            classifier,
            results_dir,
            dataset,
        )
        all_results.append(results)
    print(f"\nPlotting results for {dataset}...")
    df = pd.concat(all_results)
    df["setting_classifier"] = df["setting"] + "_" + df["classifier"]
    sns.pointplot(
        data=df, x="train_size", y="accuracy", hue="setting_classifier"
    )
    plt.axhline(y=df["dummy_accuracy"].mean(), color="k", linestyle="--")
    plt.ylabel("Accuracy")
    plt.xlabel("Training size")
    plt.savefig(os.path.join(results_dir, f"results_{start_time}.png"))
    plt.close()

    sns.boxplot(
        data=df, x="train_size", y="accuracy", hue="setting_classifier"
    )
    plt.axhline(y=df["dummy_accuracy"].mean(), color="k", linestyle="--")
    plt.ylabel("Accuracy")
    plt.xlabel("Training size")
    plt.savefig(os.path.join(results_dir, f"box_results_{start_time}.png"))
    plt.close()
