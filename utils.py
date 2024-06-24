import pandas as pd
from nilearn import maskers, image
import numpy as np
import os
from sklearn.base import clone
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
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
from nilearn.datasets import load_mni152_gm_mask

# from BBI import BlockBasedImportance
from joblib import dump, load


def _get_labels(subject, parc, nifti_dir):
    """Load labels and runs for a given subject as numpy arrays.

    Parameters
    ----------
    subject : str
        Subject identifier.
    parc : np.ndarray
        Brain data as numpy array with shape (n_samples, n_features).
        Only used to get the number of trials.
    nifti_dir : str
        directory containing the labels and runs files.

    Returns
    -------
    conditions : np.ndarray
        Trial-wise labels as numpy array.
    runs : np.ndarray
        Run number for each trial as numpy array.
    subs : np.ndarray
        A numpy array with subject identifier repeated for each trial.
    """
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


def parcellate(
    img,
    subject,
    atlas,
    data_dir,
    nifti_dir,
):
    """Convert nifti image to numpy array. If atlas is wholebrain, use
    nilearn.maskers.NiftiMasker, otherwise use nilearn.maskers.NiftiMapsMasker.

    Parameters
    ----------
    img : Nifti1Image
        Nifti image to be converted to numpy array.
    subject : str
        Subject identifier.
    atlas : sklearn.datasets.base.Bunch
        Dictionary containing the atlas maps and name.
    data_dir : str
        path to directory containing atlas/parcellation
    nifti_dir : str
        directory containing the labels and runs files.

    Returns
    -------
    data : dict
        Dictionary containing responses, conditions, runs, and subjects as
        keys, and values are corresponding numpy arrrays. Responses are the
        brain data parcellated or not depending on the atlas. Conditions are
        the trial-wise labels, runs are the run numbers for each trial, and
        subjects are the subject identifiers repeated for each trial.
    """
    data = dict(responses=[], conditions=[], runs=[], subjects=[])
    parcellate_dir = os.path.join(data_dir, atlas["name"])
    os.makedirs(parcellate_dir, exist_ok=True)
    parc_file = os.path.join(parcellate_dir, f"{subject}.npy")
    mask = load_mni152_gm_mask(resolution=3)

    if os.path.exists(parc_file):
        parc = np.load(parc_file)
    else:
        if atlas["name"] == "wholebrain":
            masker = maskers.NiftiMasker(
                mask_img=mask,
                verbose=11,
                n_jobs=20,
            )
        else:
            masker = maskers.NiftiMapsMasker(
                maps_img=atlas["maps"],
                mask_img=mask,
                verbose=11,
                n_jobs=20,
            )
        parc = masker.fit_transform(img)
        np.save(parc_file, parc)
    conditions, runs, subs = _get_labels(subject, parc, nifti_dir)
    data["responses"] = parc
    data["conditions"] = conditions
    data["runs"] = runs
    data["subjects"] = subs
    return data


def pretrain(subject, data, dummy, data_dir, atlas):
    """Pretrain a classifier for stacking on a given subject.

    Parameters
    ----------
    subject : str
        Subject identifier to pretrain the classifier.
    data : dict
        Dictionary containing responses, conditions, runs, and subjects as
        numpy arrays
    dummy : bool
        If True pretrain a dummy classifier, otherwise pretrain a linear SVC.
    data_dir : str
        path to directory to save pretrain classifiers
    atlas : sklearn.datasets.base.Bunch
        Dictionary containing the atlas maps and name.

    Returns
    -------
    tuple
        Tuple containing subject identifier and pre-trained classifier object.
    """
    pretrain_dir = os.path.join(data_dir, f"pretrain_{atlas["name"]}_l2")
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
            clf = LinearSVC(dual="auto", penalty="l1")
        # fit classifier
        clf.fit(X, Y)
        dump(clf, clf_file)
    return (f"{subject}", clf)


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
    subs_stacked=None,
    n_stacked=None,
    vary_n_stacked=False,
):
    """Fit a classifier and predict on test data.

    Parameters
    ----------
    clf : sklearn estimator
        Classifier to be fitted.
    dummy_clf : sklearn Dummy estimator
        Dummy classifier to be fitted.
    train : np.ndarray
        indices of training data.
    test : np.ndarray
        indices of test data.
    X : np.ndarray
        Brain data as numpy array with shape (n_samples, n_features).
    Y : np.ndarray
        Trial-wise labels as numpy array.
    setting : str
        Type of classifier setting, conventional or stacked (ensemble). Only
        used while saving the results for later filtering during analysis.
    n_left_out : float
        Percentage of samples left out for testing. Only used while saving the
        results for later filtering during analysis.
    classifier : str
        Name of the classifier. Only used while saving the results for later
        filtering during analysis.
    subject : str
        Subject identifier. Only used while saving the results for later
        filtering during analysis.
    subs_stacked : list, optional
        list of subjects stacked in the ensemble. Only used when we vary the
        number of subjects stacked. Saved with the results for later
        filtering during analysis, by default None
    n_stacked : int, optional
        number of stacked subjects in the ensemble. Only used while saving the
        results for later filtering during analysis, by default None
    vary_n_stacked : bool, optional
        Whether we are varying the number of subjects or not, by default False

    Returns
    -------
    dict
        Dictionary containing the results of the classification and the
        corresponding metadata.
    """
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
    if vary_n_stacked:
        result["n_stacked"] = n_stacked
        result["subs_stacked"] = subs_stacked

    return result


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
    """Main function to classify data using conventional and stacked settings.
    We keep 90% of data for training and 10\% for testing. We vary the size of
    the training set over 10 geometrically increasing sub-samples of that
    initial 90\% training split and always test the trained model on the same
    10% testing split. We do this for 20 different cross-validation train-test
    splits. Note that in the ensemble approach, while pre-training the
    classifiers, we use all the samples available in each subject.

    Parameters
    ----------
    subject : str
        Subject identifier of the current subject to decode.
    subject_i : str
        Index of the subject in the data. Used to remove the current subject
        from the pre-trained classifiers.
    data : dict
        Dictionary containing responses, conditions, runs, and subjects as
        numpy arrays.
    classifier : str
        Name of the classifier to be used for decoding. This is the classifier
        used as the final estimator in the ensemble approach. For the
        conventional approach, we use the same classifier.
    fitted_classifiers : list of tuples
        List of tuples containing the subject identifier and the pre-trained
        classifier object. Used as the estimators in the main ensemble
        approach.
    dummy_fitted_classifiers : list of tuples
        List of tuples containing the subject identifier and the pre-trained
        dummy classifier object. Used as the estimators in the dummy ensemble
        approach.
    results_dir : str
        Directory to save the results of the classification.
    dataset : str
        Name of the dataset being used for decoding.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the results of the classification.
    """
    results = []
    # select data for current subject
    sub_mask = np.where(data["subjects"] == subject)[0]
    X = data["responses"][sub_mask]
    Y = data["conditions"][sub_mask]
    groups = data["runs"][sub_mask]
    if dataset == "aomic_faces":
        cv = StratifiedShuffleSplit(
            test_size=0.20, random_state=0, n_splits=20
        )
    else:
        cv = StratifiedShuffleSplit(
            test_size=0.10, random_state=0, n_splits=20
        )
    # create conventional classifier
    if classifier == "LinearSVC":
        clf = LinearSVC(dual="auto")
    elif classifier == "RandomForest":
        clf = RandomForestClassifier(n_estimators=500, random_state=0)
    elif classifier == "MLP":
        clf = MLPClassifier(random_state=0, max_iter=1000)
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
        N = train.shape[0]
        n_classes = np.unique(Y[train]).shape[0]
        train_sizes = np.geomspace(
            n_classes * 2, N - n_classes * 2, num=10, endpoint=True
        )
        left_out_N = N - train_sizes
        left_out_percs = left_out_N / N
        for left_out in left_out_percs:
            if left_out == 0:
                train_ = train.copy()
            else:
                indices = np.arange(X.shape[0])
                train_, _ = train_test_split(
                    indices[train],
                    test_size=left_out,
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
                left_out,
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
                left_out,
                classifier,
                subject,
            )
            results.append(conventional_result)
            results.append(stacked_result)

            print(
                f"{classifier} {left_out*100:.2f}% left-out, {subject},",
                f" split {count} :",
                f"{conventional_result['balanced_accuracy']:.2f} | ",
                f"{stacked_result['balanced_accuracy']:.2f} / ",
                f"{stacked_result['dummy_balanced_accuracy']:.2f}",
            )

        count += 1

    results = pd.DataFrame(results)
    results.to_pickle(
        os.path.join(results_dir, f"results_clf_{classifier}_{subject}.pkl")
    )
    return results


def generate_sub_clf_combinations(subjects, classifiers):
    """Generate all possible combinations of subjects and classifiers. Used to
    run each subject with each classifier in parallel.

    Parameters
    ----------
    subjects : str
        List of subject identifiers.
    classifiers : str
        List of classifier names.

    Yields
    ------
    tuple
        Tuple containing the subject identifier, index of the subject
        and the classifier name.
    """
    for subject_i, subject in enumerate(subjects):
        for clf in classifiers:
            yield subject, subject_i, clf


def vary_stacked_subs(
    subject,
    subject_i,
    data,
    classifier,
    fitted_classifiers,
    dummy_fitted_classifiers,
    results_dir,
    dataset,
    how_many_n_stacked=20,
    reps_for_each_n_stacked=1,
):
    """This function is used to vary the number of subjects stacked in the
    ensemble. Here we randomly sample a subset of subjects from each dataset
    and only use the pre-trained classifiers from these subjects to train
    the final classifier. We cross-validate for each subset of subjects,
    by keeping 90% of data for training and 10\% for testing. We vary the
    size of the training set over 10 geometrically increasing sub-samples of
    that initial 90\% training split and always test the trained model on
    the same 10% testing split. We do this for 5 different cross-validation
    train-test splits, such that each split has a different subset of subjects,
    whenever possible.

    Parameters
    ----------
    subject : str
        Subject identifier of the current subject to decode.
    subject_i : str
        Index of the subject in the data. Used to remove the current subject
        from the pre-trained classifiers.
    data : dict
        Dictionary containing responses, conditions, runs, and subjects as
        numpy arrays.
    classifier : str
        Name of the classifier to be used for decoding. This is the classifier
        used as the final estimator in the ensemble approach. For the
        conventional approach, we use the same classifier.
    fitted_classifiers : list of tuples
        List of tuples containing the subject identifier and the pre-trained
        classifier object. Used as the estimators in the main ensemble
        approach.
    dummy_fitted_classifiers : list of tuples
        List of tuples containing the subject identifier and the pre-trained
        dummy classifier object. Used as the estimators in the dummy ensemble
        approach.
    results_dir : str
        Directory to save the results of the classification.
    dataset : str
        Name of the dataset being used for decoding.
    how_many_n_stacked : int, optional
        number of subjects stacked, by default 20
    reps_for_each_n_stacked : int, optional
        how many repetitions to do for a given number of subjects to be stacked
        we only use 1 rep here but can be increased to get more stable
        results for each number of subjects stacked, by default 1

    Returns
    -------
    pd.DataFrame
        DataFrame containing the results of the classification.
    """
    results = []
    # select data for current subject
    sub_mask = np.where(data["subjects"] == subject)[0]
    X = data["responses"][sub_mask]
    Y = data["conditions"][sub_mask]
    groups = data["runs"][sub_mask]
    if dataset == "aomic_faces":
        cv = StratifiedShuffleSplit(
            test_size=0.20, random_state=0, n_splits=20
        )
    else:
        cv = StratifiedShuffleSplit(test_size=0.10, random_state=0, n_splits=5)
    # create conventional classifier
    if classifier == "LinearSVC":
        clf = LinearSVC(dual="auto")
    elif classifier == "RandomForest":
        clf = RandomForestClassifier(n_estimators=500, random_state=0)
    elif classifier == "MLP":
        clf = MLPClassifier(random_state=0, max_iter=1000, verbose=11)
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
        N = train.shape[0]
        n_classes = np.unique(Y[train]).shape[0]
        train_sizes = np.geomspace(
            n_classes * 2, N - n_classes * 2, num=10, endpoint=True
        )
        left_out_N = N - train_sizes
        left_out_percs = left_out_N / N

        total_subs = len(fitted_classifiers)
        if total_subs < how_many_n_stacked:
            n_stacked_subjects = (
                np.linspace(
                    1,
                    total_subs,
                    num=total_subs - 1,
                    endpoint=False,
                )
                .round()
                .astype(int)
            )
        else:
            n_stacked_subjects = (
                np.geomspace(
                    1,
                    total_subs,
                    num=how_many_n_stacked,
                    endpoint=False,
                )
                .round()
                .astype(int)
            )
        n_stacked_subjects = np.unique(n_stacked_subjects)

        for n_stacked in n_stacked_subjects:
            # remove current subject from fitted classifiers
            fitted_classifiers_ = fitted_classifiers.copy()
            fitted_classifiers_.pop(subject_i)
            fitted_classifiers_ = np.array(fitted_classifiers_)
            dummy_fitted_classifiers_ = dummy_fitted_classifiers.copy()
            dummy_fitted_classifiers_.pop(subject_i)
            dummy_fitted_classifiers_ = np.array(dummy_fitted_classifiers_)

            # create `reps_for_each_n_stacked` random combinations of n_stacked subjects
            for rep_i in range(reps_for_each_n_stacked):
                rng = np.random.default_rng()
                picked_subjects = rng.choice(
                    len(fitted_classifiers_),
                    int(n_stacked),
                    replace=False,
                )
                # select a subset of fitted classifiers
                fitted_classifiers__ = fitted_classifiers_[picked_subjects]
                dummy_fitted_classifiers__ = dummy_fitted_classifiers_[
                    picked_subjects
                ]
                subs_stacked = [clf[0] for clf in fitted_classifiers__]
                for left_out in left_out_percs:
                    if left_out == 0:
                        train_ = train.copy()
                    else:
                        indices = np.arange(X.shape[0])
                        train_, _ = train_test_split(
                            indices[train],
                            test_size=left_out,
                            random_state=0,
                            stratify=Y[train],
                        )

                    # create stacked classifier
                    stacked_clf = StackingClassifier(
                        list(fitted_classifiers__),
                        final_estimator=clone(clf),
                        cv="prefit",
                    )
                    dummy_stacked_clf = StackingClassifier(
                        list(dummy_fitted_classifiers__),
                        final_estimator=DummyClassifier(
                            strategy="most_frequent"
                        ),
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
                        left_out,
                        classifier,
                        subject,
                        subs_stacked,
                        n_stacked,
                        vary_n_stacked=True,
                    )
                    results.append(stacked_result)

                    print(
                        f"{classifier} {left_out*100:.2f}% left-out, {subject}, split {count},",
                        f"{n_stacked} subs stacked, rep {rep_i} : {stacked_result['balanced_accuracy']:.2f} / {stacked_result['dummy_balanced_accuracy']:.2f}",
                    )
        count += 1

    results = pd.DataFrame(results)
    results.to_pickle(
        os.path.join(results_dir, f"results_clf_{classifier}_{subject}.pkl")
    )
    return results


def feature_importance(
    subject,
    subject_i,
    data,
    classifier,
    fitted_classifiers,
    dummy_fitted_classifiers,
    results_dir,
    dataset,
    n_jobs,
):
    """Compute feature importance using the BlockBasedImportance method.
    We only compute the feature importance for the stacked classifier, using
    DiFuMo features and the Random Forest classiifer as the final classifier
    (which is the only one we change, the first classifier is still LinearSVC).
    These scores correspond to DiFuMo features but can later be projected to
    the full-voxel feature space.

    Note that this computation is independent of the other decoding
    experiments, the BlockBasedImportance estimator internally fits the
    decoding object and then computes the importance scores.

    Parameters
    ----------
    subject : str
        Subject identifier for the current subject to do decode and then get
        the importances for.
    subject_i : str
        Index of the current subject
    data : dict
        Dictionary containing responses, conditions, runs, and subjects as
        numpy arrays.
    classifier : str
        Name of the classifier to be used for decoding. This is the classifier
        used as the final estimator in the ensemble approach. Only works with
        RandomForest as of now.
    fitted_classifiers : list of tuples
        List of tuples containing the subject identifier and the pre-trained
        classifier object.
    dummy_fitted_classifiers : list of tuples
        List of tuples containing the subject identifier and the pre-trained
        dummy classifier object.
    results_dir : str
        Directory to save the results of the classification.
    dataset : str
        Name of the dataset being used for decoding.
    n_jobs : int
        Number of jobs to run in parallel for the BlockBasedImportance method.

    Returns
    -------
    dict
        Dictionary containing the feature importance scores for the stacked
    """
    # select data for current subject
    sub_mask = np.where(data["subjects"] == subject)[0]
    X = data["responses"][sub_mask]
    Y = data["conditions"][sub_mask]

    # create conventional classifier
    if classifier == "LinearSVC":
        clf = LinearSVC(dual="auto")
    elif classifier == "RandomForest":
        clf = RandomForestClassifier(n_estimators=500, random_state=0)
    elif classifier == "MLP":
        clf = MLPClassifier(random_state=0, max_iter=1000)

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
    # importance estimator
    stacked_bbi_model = BlockBasedImportance(
        estimator=stacked_clf,
        prob_type="classification",
        n_jobs=n_jobs,
        random_state=0,
        verbose=0,
        do_hyper=False,
    )
    stacked_bbi_model.fit(X, Y)
    stacked_importance = stacked_bbi_model.compute_importance()
    stacked_importance["subject"] = subject
    stacked_importance["classifier"] = classifier
    stacked_importance["setting"] = "stacked"
    stacked_importance["dataset"] = dataset

    dump(
        stacked_importance,
        os.path.join(results_dir, f"featimp_clf_{classifier}_{subject}.pkl"),
    )
    return stacked_importance
