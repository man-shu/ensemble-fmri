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
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Ridge,
    Lasso,
)
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import LinearSVC


### classification function ###
def _classify(
    clf,
    dummy_clf,
    train,
    test,
    X,
    Y,
    setting,
    n_left_out,
    classifier,
    subject,
):
    result = {}
    clf.fit(X[train], Y[train])
    dummy_clf.fit(X[train], Y[train])
    prediction = clf.predict(X[test])
    dummy_prediction = dummy_clf.predict(X[test])
    accuracy = clf.score(X[test], Y[test])
    dummy_accuracy = dummy_clf.score(X[test], Y[test])
    result["accuracy"] = accuracy
    result["dummy_accuracy"] = dummy_accuracy
    result["subject"] = subject
    result["true"] = Y[test]
    result["predicted"] = prediction
    result["dummy_predicted"] = dummy_prediction
    result["left_out"] = n_left_out
    result["train_size"] = len(train)
    result["setting"] = setting
    result["classifier"] = classifier

    return result


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
    cv = ShuffleSplit(test_size=0.20, random_state=0, n_splits=10)
    # create conventional classifier
    if classifier == "LogisticRegression":
        clf = LogisticRegression()
    elif classifier == "LinearRegression":
        clf = LinearRegression()
    elif classifier == "Ridge":
        clf = Ridge()
    elif classifier == "MLPRegressor":
        clf = MLPRegressor()
    elif classifier == "Lasso":
        clf = Lasso()
    elif classifier == "MultiLabelSVC":
        clf = MultiOutputClassifier(LinearSVC(dual="auto"), n_jobs=20)
    elif classifier == "MLPClassifier":
        clf = MLPClassifier(verbose=11, max_iter=int(1e5), tol=1e-6)
    else:
        raise ValueError("Classifier not implemented.")
    if classifier == "MultiLabelSVC":
        dummy_clf = MultiOutputClassifier(DummyClassifier(), n_jobs=20)
    elif classifier == "MLPClassifier":
        dummy_clf = DummyClassifier()
    else:
        dummy_clf = DummyRegressor()
    count = 0
    # utils._plot_cv_indices(
    #     cv,
    #     X,
    #     Y,
    #     groups,
    #     subject,
    #     5,
    #     results_dir,
    # )
    for train, test in tqdm(
        cv.split(X, Y, groups=groups),
        desc=f"{dataset}, {subject}, {classifier}",
        position=0,
        leave=True,
        total=cv.get_n_splits(),
    ):
        conventional_result = _classify(
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
        results.append(conventional_result)
        print(
            f"{classifier} {20}% left-out, {subject}, split {count} :",
            f"{conventional_result['accuracy']:.2f} / {conventional_result['dummy_accuracy']:.2f}",
        )

        count += 1

    results = pd.DataFrame(results)
    results.to_pickle(
        os.path.join(results_dir, f"results_clf_{classifier}_{subject}.pkl")
    )
    return results


### get class and group labels ###
def _get_labels(subject, parc, nifti_dir):
    # get class labels
    label_file = glob(os.path.join(nifti_dir, f"{subject}_multi_labels*"))[0]
    _, label_ext = os.path.splitext(label_file)
    if label_ext == ".csv":
        conditions = pd.read_csv(label_file, header=None)
        conditions = conditions[0].values
    elif label_ext == ".npy":
        conditions = np.load(label_file, allow_pickle=True)
    # get run labels
    run_file = glob(os.path.join(nifti_dir, f"{subject}_runs*"))
    if len(run_file) == 0:
        runs = np.ones_like(conditions)
    else:
        run_file = run_file[0]
        _, run_ext = os.path.splitext(run_file)
        if run_ext == ".csv":
            runs = pd.read_csv(run_file, header=None)
            runs = runs[0].values
        elif run_ext == ".npy":
            runs = np.load(run_file, allow_pickle=True)
    # get number of trials
    num_trials = parc.shape[0]
    subs = np.repeat(subject, num_trials)
    return conditions, runs, subs


### parcellate data ###
def parcellate(
    img,
    subject,
    atlas,
    DATA_ROOT,
    data_dir,
    nifti_dir,
):
    data = dict(responses=[], conditions=[], runs=[], subjects=[])
    parcellate_dir = os.path.join(data_dir, f"{atlas.name}_standardized")
    os.makedirs(parcellate_dir, exist_ok=True)
    parc_file = os.path.join(parcellate_dir, f"{subject}.npy")
    if os.path.exists(parc_file):
        parc = np.load(parc_file)
    else:
        masker = maskers.NiftiMapsMasker(
            maps_img=atlas.maps,
            memory=DATA_ROOT,
            verbose=11,
            n_jobs=20,
            standardize=True,
        )
        parc = masker.fit_transform(img)
        np.save(parc_file, parc)
    conditions, runs, subs = _get_labels(subject, parc, nifti_dir)
    data["responses"] = parc
    data["conditions"] = conditions.astype(int)
    data["runs"] = runs
    data["subjects"] = subs
    return data


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
    classifier = "MLPClassifier"

    # input data root path
    data_dir = os.path.join(DATA_ROOT, dataset)
    data_resolution = "3mm"  # or 1_5mm
    nifti_dir = os.path.join(data_dir, data_resolution)

    # get difumo atlas
    atlas = datasets.fetch_atlas_difumo(
        dimension=1024, resolution_mm=3, data_dir=DATA_ROOT
    )
    atlas["name"] = "difumo"

    # output results path
    start_time = time.strftime("%Y%m%d-%H%M%S")
    results_dir = f"{dataset}_{atlas.name}_{classifier}_results_{start_time}"
    results_dir = os.path.join(OUT_ROOT, results_dir)
    os.makedirs(results_dir, exist_ok=True)

    # get file names
    imgs = glob(os.path.join(nifti_dir, "*.nii.gz"))
    subjects = [os.path.basename(img).split(".")[0] for img in imgs]

    print(f"\nParcellating {dataset}...")
    data = Parallel(n_jobs=2, verbose=2, backend="loky")(
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

    print(f"\nRunning cross-val on {dataset}...")
    all_results = Parallel(n_jobs=len(subjects), verbose=11, backend="loky")(
        delayed(cross_val)(
            subject,
            data,
            classifier,
            results_dir,
            dataset,
        )
        for subject in subjects
    )

    print(f"\nPlotting results for {dataset}...")
    df = pd.concat(all_results)
    df["setting_classifier"] = df["setting"] + "_" + df["classifier"]
    sns.barplot(
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
