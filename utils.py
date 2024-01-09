import pandas as pd
from nilearn import maskers, image
import numpy as np
import os
from sklearn.base import clone
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import (
    ShuffleSplit,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from glob import glob


### get class and group labels ###
def _get_labels(subject, parc, nifti_dir):
    # get class labels
    label_file = glob(os.path.join(nifti_dir, f"{subject}_labels*"))[0]
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
    parcellate_dir = os.path.join(data_dir, atlas.name)
    os.makedirs(parcellate_dir, exist_ok=True)
    parc_file = os.path.join(parcellate_dir, f"{subject}.npy")
    if os.path.exists(parc_file):
        parc = np.load(parc_file)
    else:
        if atlas.name == "wholebrain":
            ref_img = image.index_img(img, 0)
            ref_img_shape = ref_img.shape
            wholebrain_mask = image.new_img_like(
                ref_img, np.ones(ref_img_shape)
            )
            masker = maskers.NiftiMasker(
                mask_img=wholebrain_mask,
                memory=DATA_ROOT,
                verbose=11,
                n_jobs=20,
            )
        else:
            masker = maskers.NiftiMapsMasker(
                maps_img=atlas.maps, memory=DATA_ROOT, verbose=11, n_jobs=20
            )
        parc = masker.fit_transform(img)
        np.save(parc_file, parc)
    conditions, runs, subs = _get_labels(subject, parc, nifti_dir)
    data["responses"] = parc
    data["conditions"] = conditions
    data["runs"] = runs
    data["subjects"] = subs
    return data


#### pretraining for stacking ####
def pretrain(subject, data, dummy, data_dir, atlas):
    pretrain_dir = os.path.join(data_dir, f"pretrain_{atlas.name}")
    os.makedirs(pretrain_dir, exist_ok=True)
    if dummy:
        file_id = "dummy"
    else:
        file_id = "linear"
    clf_file = os.path.join(pretrain_dir, f"{subject}_{file_id}.pkl")
    if os.path.exists(clf_file):
        clf = load(clf_file)
    else:
        # select data for current subject
        sub_mask = np.where(data["subjects"] == subject)[0]
        X = data["responses"][sub_mask]
        Y = data["conditions"][sub_mask]
        if dummy:
            clf = DummyClassifier(strategy="most_frequent")
        else:
            clf = LinearSVC(dual="auto")
        # fit classifier
        clf.fit(X, Y)
        dump(clf, clf_file)
    return (f"{subject}", clf)


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
    accuracy = accuracy_score(Y[test], prediction)
    dummy_accuracy = accuracy_score(Y[test], dummy_prediction)
    result["accuracy"] = accuracy
    result["dummy_accuracy"] = dummy_accuracy
    result["balanced_accuracy"] = balanced_accuracy_score(Y[test], prediction)
    result["dummy_balanced_accuracy"] = balanced_accuracy_score(
        Y[test], dummy_prediction
    )
    result["subject"] = subject
    result["true"] = Y[test]
    result["predicted"] = prediction
    result["dummy_predicted"] = dummy_prediction
    result["left_out"] = n_left_out
    result["train_size"] = len(train)
    result["setting"] = setting
    result["classifier"] = classifier

    return result


### plot cv splits ###
def _plot_cv_indices(
    cv,
    X,
    y,
    group,
    subject,
    n_splits,
    out_dir,
    lw=10,
):
    """Create a sample plot for indices of a cross-validation object."""
    fig, ax = plt.subplots()
    cmap_data = plt.cm.tab20
    cmap_cv = plt.cm.coolwarm
    _, y = np.unique(y, return_inverse=True)
    _, group = np.unique(group, return_inverse=True)
    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X, y, group)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0
        # Visualize the results
        ax.scatter(
            range(len(indices)),
            [ii + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=lw,
            cmap=cmap_cv,
            vmin=-0.2,
            vmax=1.2,
        )
    # Plot the data classes and groups at the end
    ax.scatter(
        range(len(X)),
        [ii + 1.5] * len(X),
        c=y,
        marker="_",
        lw=lw,
        cmap=cmap_data,
    )
    ax.scatter(
        range(len(X)),
        [ii + 2.5] * len(X),
        c=group,
        marker="_",
        lw=lw,
        cmap=cmap_data,
    )
    # Formatting
    yticklabels = [*range(n_splits)] + ["class", "group"]
    ax.set(
        yticks=np.arange(n_splits + 2) + 0.5,
        yticklabels=yticklabels,
        xlabel="Sample index",
        ylabel="CV iteration",
        ylim=[n_splits + 2.2, -0.2],
    )
    split_dir = os.path.join(out_dir, "test_train_splits")
    os.makedirs(split_dir, exist_ok=True)
    ax.set_title(f"Train/test splits with {subject}% of samples left-out")
    plot_file = f"{subject}_cv_indices.png"
    plot_file = os.path.join(split_dir, plot_file)
    fig.savefig(plot_file, bbox_inches="tight")
    plt.close()


### cross-validation ###
def decode(
    subject,
    subject_i,
    data,
    classifier,
    fitted_classifiers,
    dummy_fitted_classifiers,
    results_dir,
    dataset,
):
    results = []
    # select data for current subject
    sub_mask = np.where(data["subjects"] == subject)[0]
    X = data["responses"][sub_mask]
    Y = data["conditions"][sub_mask]
    groups = data["runs"][sub_mask]
    cv = StratifiedShuffleSplit(test_size=0.10, random_state=0, n_splits=20)
    # create conventional classifier
    if classifier == "LinearSVC":
        clf = LinearSVC(dual="auto")
    elif classifier == "RandomForest":
        clf = RandomForestClassifier(n_estimators=200, random_state=0)
    count = 0
    _plot_cv_indices(
        cv,
        X,
        Y,
        groups,
        subject,
        20,
        results_dir,
    )
    for train, test in tqdm(
        cv.split(X, Y, groups=groups),
        desc=f"{dataset}, {subject}, {classifier}",
        position=0,
        leave=True,
        total=cv.get_n_splits(),
    ):
        for n_left_out in range(0, 100, 10):
            if n_left_out == 0:
                train_ = train.copy()
            else:
                indices = np.arange(X.shape[0])
                train_, _ = train_test_split(
                    indices[train],
                    test_size=n_left_out / 100,
                    random_state=0,
                    stratify=Y[train],
                )
            clf = clone(clf)
            dummy_clf = DummyClassifier(strategy="most_frequent")
            conventional_result = _classify(
                clf,
                dummy_clf,
                train_,
                test,
                X,
                Y,
                "conventional",
                n_left_out,
                classifier,
                subject,
            )
            # remove current subject from fitted classifiers
            fitted_classifiers_ = fitted_classifiers.copy()
            fitted_classifiers_.pop(subject_i)
            dummy_fitted_classifiers_ = dummy_fitted_classifiers.copy()
            dummy_fitted_classifiers_.pop(subject_i)
            # create stacked classifier
            stacked_clf = StackingClassifier(
                fitted_classifiers_,
                final_estimator=clone(clf),
                cv="prefit",
            )
            dummy_stacked_clf = StackingClassifier(
                dummy_fitted_classifiers_,
                final_estimator=DummyClassifier(strategy="most_frequent"),
                cv="prefit",
            )
            stacked_result = _classify(
                stacked_clf,
                dummy_stacked_clf,
                train_,
                test,
                X,
                Y,
                "stacked",
                n_left_out,
                classifier,
                subject,
            )
            results.append(conventional_result)
            results.append(stacked_result)

            print(
                f"{classifier} {n_left_out}% left-out, {subject}, split {count} :",
                f"{conventional_result['balanced_accuracy']:.2f} | {stacked_result['balanced_accuracy']:.2f} / {stacked_result['dummy_balanced_accuracy']:.2f}",
            )

        count += 1

    results = pd.DataFrame(results)
    results.to_pickle(
        os.path.join(results_dir, f"results_clf_{classifier}_{subject}.pkl")
    )
    return results


### generator for subject classifier combinations ###
def generate_sub_clf_combinations(subjects, classifiers):
    for subject_i, subject in enumerate(subjects):
        for clf in classifiers:
            yield subject, subject_i, clf
