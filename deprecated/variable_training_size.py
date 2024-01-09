import pandas as pd
from nilearn import image
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import (
    GroupShuffleSplit,
    LeavePGroupsOut,
    LeaveOneGroupOut,
    StratifiedShuffleSplit,
    ShuffleSplit,
)
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import ibc_public.utils_data
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import time
import pickle
from sklearn.ensemble import StackingClassifier

# input data root path
data_dir = "rsvp_trial"
data_resolution = "3mm"  # or 1_5mm
nifti_dir = os.path.join(data_dir, data_resolution)

# output results path
start_time = time.strftime("%Y%m%d-%H%M%S")
results_dir = f"bench_results_{start_time}"
os.makedirs(results_dir, exist_ok=True)

# trial conditions
conditions = pd.read_csv(os.path.join(data_dir, "labels.csv"), header=None)
conditions = conditions[0].values
# run labels
runs = pd.read_csv(os.path.join(data_dir, "runs.csv"), header=None)
runs = runs[0].values
n_runs = len(np.unique(runs))
# get subject list for rsvp-language
subjects_sessions = ibc_public.utils_data.get_subject_session("rsvp-language")
subjects = np.array(
    [subject for subject, session in subjects_sessions], dtype="object"
)
subjects = np.unique(subjects)
# create empty dictionary to store data
data = dict(responses=[], conditions=[], runs=[], subjects=[])
# load data
print("Loading data...")
for subject in tqdm(subjects):
    try:
        response = image.load_img(os.path.join(nifti_dir, f"{subject}.nii.gz"))
        response.shape
    except:
        print(f"{subject} not found")
        continue
    response = response.get_fdata()
    checksum = response[:, :, :, 0].sum().round(5)
    # reshape from (53, 63, 52, 360) to 2d array (360, 53*63*52)
    response = response.reshape(-1, response.shape[-1]).T
    assert response[0, :].sum().round(5) == checksum
    # get number of trials
    num_trials = response.shape[0]
    subs = np.repeat(subject, num_trials)
    # append to dictionary
    data["responses"].append(response)
    data["conditions"].append(conditions)
    data["runs"].append(runs)
    data["subjects"].append(subs)

print("Concatenating data...")
# concatenate data
for dat in ["responses", "conditions", "runs", "subjects"]:
    data[dat] = np.concatenate(data[dat])

subjects = np.unique(data["subjects"])

#### pretraining for stacking ####
fitted_classifiers = []
dummy_fitted_classifiers = []
for subject in tqdm(subjects):
    clf = LinearSVC(dual="auto")
    dummy_clf = DummyClassifier(strategy="most_frequent")
    # select data for current subject
    sub_mask = np.where(data["subjects"] == subject)[0]
    X = data["responses"][sub_mask]
    y = data["conditions"][sub_mask]
    # fit classifier
    clf.fit(X, y)
    dummy_clf.fit(X, y)
    # save classifier
    fitted_classifiers.append((f"{subject}", clf))
    dummy_fitted_classifiers.append((f"{subject}", dummy_clf))


# classification function
def classify(clf, dummy_clf, train, test, X, Y, setting, count, n_left_out):
    result = {}
    clf = clf.fit(X[train], Y[train])
    dummy_clf = dummy_clf.fit(X[train], Y[train])
    prediction = clf.predict(X[test])
    dummy_prediction = dummy_clf.predict(X[test])
    accuracy = accuracy_score(Y[test], prediction)
    dummy_accuracy = accuracy_score(Y[test], dummy_prediction)
    result["accuracy"] = accuracy
    result["dummy_accuracy"] = dummy_accuracy
    result["subject"] = subject
    result["true"] = y[test]
    result["predicted"] = prediction
    result["dummy_predicted"] = dummy_prediction
    result["left_out"] = n_left_out
    result["train_size"] = len(train)
    result["setting"] = setting

    return result


#### cross-validation across runs ####
results = []
# vary number of runs left out for testing
for n_left_out in tqdm(range(1, n_runs)):
    print(f"Leaving {n_left_out} runs out")
    cv = LeavePGroupsOut(n_left_out)
    for sub_i, subject in enumerate(subjects):
        print(f"Subject {subject}")
        # select data for current subject
        sub_mask = np.where(data["subjects"] == subject)[0]
        X = data["responses"][sub_mask]
        Y = data["conditions"][sub_mask]
        groups = data["runs"][sub_mask]
        # remove current subject from fitted classifiers
        fitted_classifiers_ = fitted_classifiers.copy()
        fitted_classifiers_.pop(sub_i)
        dummy_fitted_classifiers_ = dummy_fitted_classifiers.copy()
        dummy_fitted_classifiers_.pop(sub_i)
        # create stacked classifier
        stacked_clf = StackingClassifier(
            fitted_classifiers_,
            final_estimator=LinearSVC(dual="auto"),
            cv="prefit",
        )
        dummy_stacked_clf = StackingClassifier(
            dummy_fitted_classifiers_,
            final_estimator=DummyClassifier(strategy="most_frequent"),
            cv="prefit",
        )
        # create conventional classifier
        clf = LinearSVC(dual="auto")
        dummy_clf = DummyClassifier(strategy="most_frequent")
        count = 0
        for train, test in cv.split(X, Y, groups=groups):
            conventional_result = classify(
                clf, dummy_clf, train, test, X, Y, "conventional", count
            )
            stacked_result = classify(
                stacked_clf,
                dummy_stacked_clf,
                train,
                test,
                X,
                Y,
                "stacked",
                count,
            )
            results.append(conventional_result)
            results.append(stacked_result)
            count += 1

results = pd.DataFrame(results)
results.to_pickle(os.path.join(results_dir, "results.pkl"))

#### plot results ####
sns.boxplot(data=results, x="train_size", y="accuracy", hue="setting")
plt.axhline(y=results["dummy_accuracy"].mean(), color="k", linestyle="--")
plt.ylabel("Accuracy")
plt.xlabel("Training size")
plt.title("Across runs")
plt.savefig(f"results_{time.strftime('%Y%m%d-%H%M%S')}.png")
plt.close()


#### cross-validation within each run ####
results = []
# vary percentage of testing data
for n_left_out in tqdm(range(10, 100, 10)):
    print(f"Leaving {n_left_out}% of trials out")
    cv = StratifiedShuffleSplit(
        test_size=n_left_out / 100, random_state=0, n_splits=30
    )
    for sub_i, subject in enumerate(subjects):
        # remove current subject from fitted classifiers
        fitted_classifiers_ = fitted_classifiers.copy()
        fitted_classifiers_.pop(sub_i)
        dummy_fitted_classifiers_ = dummy_fitted_classifiers.copy()
        dummy_fitted_classifiers_.pop(sub_i)
        # create stacked classifier
        stacked_clf = StackingClassifier(
            fitted_classifiers_,
            final_estimator=LinearSVC(dual="auto"),
            cv="prefit",
        )
        dummy_stacked_clf = StackingClassifier(
            dummy_fitted_classifiers_,
            final_estimator=DummyClassifier(strategy="most_frequent"),
            cv="prefit",
        )
        # create conventional classifier
        clf = LinearSVC(dual="auto")
        dummy_clf = DummyClassifier(strategy="most_frequent")
        for run in range(n_runs):
            print(f"Subject {subject}, Run 0{run}")
            # select data for current sub, run
            sub_run_mask = np.where(
                (data["subjects"] == subject) & (data["runs"] == run)
            )[0]
            X = data["responses"][sub_run_mask]
            Y = data["conditions"][sub_run_mask]
            count = 0
            for train, test in cv.split(X, Y):
                conventional_result = classify(
                    clf, dummy_clf, train, test, X, Y, "conventional", count
                )
                conventional_result["run"] = run
                stacked_result = classify(
                    stacked_clf,
                    dummy_stacked_clf,
                    train,
                    test,
                    X,
                    Y,
                    "stacked",
                    count,
                )
                stacked_result["run"] = run
                results.append(conventional_result)
                results.append(stacked_result)
                count += 1

results = pd.DataFrame(results)
results.to_pickle(os.path.join(results_dir, "results.pkl"))

#### plot results ####
sns.boxplot(data=results, x="train_size", y="accuracy", hue="setting")
sns.pointplot(data=results, x="train_size", y="accuracy", hue="setting")
plt.axhline(y=results["dummy_accuracy"].mean(), color="k", linestyle="--")
plt.ylabel("Accuracy")
plt.xlabel("Training size")
# plt.title("Within run")
plt.savefig(
    f"bench_results_20231004-132128/box_results_{time.strftime('%Y%m%d-%H%M%S')}.png"
)
plt.close()


train_sizes = np.unique(results["train_size"])
dfs = []
for train_size in train_sizes:
    for setting in ["conventional", "stacked"]:
        sub_results = results[
            (results["train_size"] == train_size)
            & (results["setting"] == setting)
        ]
        df = {}
        true = []
        predicted = []
        dummy_predicted = []
        for _, row in sub_results.iterrows():
            true.extend(row["true"])
            predicted.extend(row["predicted"])
            dummy_predicted.extend(row["dummy_predicted"])
        df["accuracy"] = accuracy_score(true, predicted)
        df["dummy_accuracy"] = accuracy_score(true, dummy_predicted)
        df["setting"] = setting
        df["train_size"] = train_size

        dfs.append(df)

dfs = pd.DataFrame(dfs)
sns.boxplot(data=dfs, x="train_size", y="accuracy", hue="setting")
plt.axhline(y=results["dummy_accuracy"].mean(), color="k", linestyle="--")
plt.ylabel("Accuracy")
plt.xlabel("Training size")
# plt.title("Within run")
plt.savefig(
    f"bench_results_20231004-132128/compiled_box_results_{time.strftime('%Y%m%d-%H%M%S')}.png"
)
plt.close()

sns.pointplot(data=dfs, x="train_size", y="accuracy", hue="setting")
plt.axhline(y=results["dummy_accuracy"].mean(), color="k", linestyle="--")
plt.ylabel("Accuracy")
plt.xlabel("Training size")
# plt.title("Within run")
plt.savefig(
    f"bench_results_20231004-132128/compiled_point_results_{time.strftime('%Y%m%d-%H%M%S')}.png"
)
plt.close()


def _plot_cv_indices(
    cv,
    X,
    y,
    group,
    n_left_out,
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
    ax.set_title(f"Train/test splits with {n_left_out}% of samples left-out")
    plot_file = f"{n_left_out}_cv_indices.png"
    plot_file = os.path.join(split_dir, plot_file)
    fig.savefig(plot_file, bbox_inches="tight")
    plt.close()


def over_all_runs(n_left_out, subjects, data):
    results = []
    cv = ShuffleSplit(test_size=n_left_out / 100, random_state=0, n_splits=20)
    for sub_i, subject in tqdm(
        enumerate(subjects),
        desc=f"{n_left_out}% left-out",
        position=0,
        leave=True,
        total=len(subjects),
    ):
        print(f"Subject {subject}")
        # select data for current subject
        sub_mask = np.where(data["subjects"] == subject)[0]
        X = data["responses"][sub_mask]
        Y = data["conditions"][sub_mask]
        groups = data["runs"][sub_mask]
        # remove current subject from fitted classifiers
        fitted_classifiers_ = fitted_classifiers.copy()
        fitted_classifiers_.pop(sub_i)
        dummy_fitted_classifiers_ = dummy_fitted_classifiers.copy()
        dummy_fitted_classifiers_.pop(sub_i)
        # create stacked classifier
        stacked_clf = StackingClassifier(
            fitted_classifiers_,
            final_estimator=LinearSVC(dual="auto"),
            cv="prefit",
        )
        dummy_stacked_clf = StackingClassifier(
            dummy_fitted_classifiers_,
            final_estimator=DummyClassifier(strategy="most_frequent"),
            cv="prefit",
        )
        # create conventional classifier
        clf = LinearSVC(dual="auto")
        dummy_clf = DummyClassifier(strategy="most_frequent")
        count = 0
        _plot_cv_indices(
            cv,
            X,
            Y,
            groups,
            n_left_out,
            20,
            results_dir,
        )
        for train, test in cv.split(X, Y, groups=groups):
            conventional_result = classify(
                clf,
                dummy_clf,
                train,
                test,
                X,
                Y,
                "conventional",
                count,
                n_left_out,
            )
            stacked_result = classify(
                stacked_clf,
                dummy_stacked_clf,
                train,
                test,
                X,
                Y,
                "stacked",
                count,
                n_left_out,
            )
            results.append(conventional_result)
            results.append(stacked_result)

            print(
                f" {n_left_out}% left-out, {subject}, split {count} :",
                f"{conventional_result['accuracy']:.2f} | {stacked_result['accuracy']:.2f} / {conventional_result['dummy_accuracy']:.2f}",
            )

            count += 1

    results = pd.DataFrame(results)
    results.to_pickle(
        os.path.join(results_dir, f"results_{n_left_out}_leftout.pkl")
    )
    return results


# vary number of samples left out for testing
all_results = Parallel(n_jobs=6, verbose=2, backend="loky")(
    delayed(over_all_runs)(n_left_out, subjects, data)
    for n_left_out in range(10, 100, 10)
)
df = pd.concat(all_results)
sns.pointplot(data=df, x="train_size", y="accuracy", hue="setting")
plt.axhline(y=df["dummy_accuracy"].mean(), color="k", linestyle="--")
plt.ylabel("Accuracy")
plt.xlabel("Training size")
# plt.title("Within run")
plt.savefig(os.path.join(results_dir, f"results_{start_time}.png"))
plt.close()

sns.boxplot(data=df, x="train_size", y="accuracy", hue="setting")
plt.axhline(y=df["dummy_accuracy"].mean(), color="k", linestyle="--")
plt.ylabel("Accuracy")
plt.xlabel("Training size")
# plt.title("Within run")
plt.savefig(os.path.join(results_dir, f"box_results_{start_time}.png"))
plt.close()


# dimensionality reduction
def reduce(X):
    from sklearn.decomposition import PCA
    from sklearn.decomposition import FastICA
    from sklearn.decomposition import MiniBatchDictionaryLearning

    # pca = PCA()
    # ica = FastICA()
    dl = MiniBatchDictionaryLearning(verbose=True)
    # X_reduced = pca.fit_transform(X)
    # X_reduced = ica.fit_transform(X)
    X_reduced = dl.fit_transform(X)

    return X_reduced


def pca(n_left_out, subjects, data):
    results = []
    cv = ShuffleSplit(test_size=n_left_out / 100, random_state=0, n_splits=20)
    for sub_i, subject in tqdm(
        enumerate(subjects),
        desc=f"{n_left_out}% left-out",
        position=0,
        leave=True,
        total=len(subjects),
    ):
        print(f"Subject {subject}")
        # select data for current subject
        sub_mask = np.where(data["subjects"] == subject)[0]
        X = data["responses"][sub_mask]
        Y = data["conditions"][sub_mask]
        groups = data["runs"][sub_mask]
        # reduce dimensionality
        X = reduce(X)
        # create conventional classifier
        clf = LinearSVC(dual="auto")
        dummy_clf = DummyClassifier(strategy="most_frequent")
        count = 0
        _plot_cv_indices(
            cv,
            X,
            Y,
            groups,
            n_left_out,
            20,
            results_dir,
        )
        for train, test in cv.split(X, Y, groups=groups):
            pca_result = classify(
                clf,
                dummy_clf,
                train,
                test,
                X,
                Y,
                "dictionary_learning",
                count,
                n_left_out,
            )
            results.append(pca_result)

            print(
                f" {n_left_out}% left-out, {subject}, split {count} :",
                f"{pca_result['accuracy']:.2f} / {pca_result['dummy_accuracy']:.2f}",
            )

            count += 1

    results = pd.DataFrame(results)
    results.to_pickle(
        os.path.join(results_dir, f"results_{n_left_out}_leftout.pkl")
    )
    return results


# vary number of samples left out for testing
all_pca_results = Parallel(n_jobs=9, verbose=2, backend="loky")(
    delayed(pca)(n_left_out, subjects, data)
    for n_left_out in range(10, 100, 10)
)
all_pca_results.extend(all_results)
df = pd.concat(all_pca_results)

sns.pointplot(data=df, x="train_size", y="accuracy", hue="setting")
plt.axhline(y=df["dummy_accuracy"].mean(), color="k", linestyle="--")
plt.ylabel("Accuracy")
plt.xlabel("Training size")
# plt.title("Within run")
plt.savefig(
    os.path.join(results_dir, f"results_{time.strftime('%Y%m%d-%H%M%S')}.png")
)
plt.close()

sns.boxplot(data=df, x="train_size", y="accuracy", hue="setting")
plt.axhline(y=df["dummy_accuracy"].mean(), color="k", linestyle="--")
plt.ylabel("Accuracy")
plt.xlabel("Training size")
# plt.title("Within run")
plt.savefig(
    os.path.join(
        results_dir, f"box_results_{time.strftime('%Y%m%d-%H%M%S')}.png"
    )
)
plt.close()
