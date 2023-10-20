import pandas as pd
from nilearn import image
import numpy as np
import os
from sklearn.base import clone
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import (
    ShuffleSplit,
    train_test_split,
)
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import time
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
import ibc_public.utils_data


# input data root path
data_dir = "rsvp_trial"
data_resolution = "3mm"  # or 1_5mm
nifti_dir = os.path.join(data_dir, data_resolution)
classifiers = ["LinearSVC", "RandomForest"]

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
# create empty dictionary to store data
data = dict(responses=[], conditions=[], runs=[], subjects=[])
# load data
print("Loading data...")
for subject in tqdm(subjects):
    try:
        response = image.load_img(os.path.join(nifti_dir, f"{subject}.nii.gz"))
        print(response.shape)
    except Exception as error:
        print(error)
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
print("Done!")

print("Concatenating data...")
# concatenate data
for dat in ["responses", "conditions", "runs", "subjects"]:
    data[dat] = np.concatenate(data[dat])
print("Done!")

subjects = np.unique(data["subjects"])

print("Pretraining classifiers...")
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
print("Done!")


# classification function
def classify(
    clf, dummy_clf, train, test, X, Y, setting, count, n_left_out, classifier
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
    result["subject"] = subject
    result["true"] = y[test]
    result["predicted"] = prediction
    result["dummy_predicted"] = dummy_prediction
    result["left_out"] = n_left_out
    result["train_size"] = len(train)
    result["setting"] = setting
    result["classifier"] = classifier

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
):
    results = []
    # select data for current subject
    sub_mask = np.where(data["subjects"] == subject)[0]
    X = data["responses"][sub_mask]
    Y = data["conditions"][sub_mask]
    groups = data["runs"][sub_mask]
    cv = ShuffleSplit(test_size=0.10, random_state=0, n_splits=20)
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
        desc=f"{subject}, {classifier}",
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
            conventional_result = classify(
                clf,
                dummy_clf,
                train_,
                test,
                X,
                Y,
                "conventional",
                count,
                n_left_out,
                classifier,
            )
            # remove current subject from fitted classifiers
            fitted_classifiers_ = fitted_classifiers.copy()
            fitted_classifiers_.pop(subject_i)
            print(fitted_classifiers_)
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
            stacked_result = classify(
                stacked_clf,
                dummy_stacked_clf,
                train_,
                test,
                X,
                Y,
                "stacked",
                count,
                n_left_out,
                classifier,
            )
            results.append(conventional_result)
            results.append(stacked_result)

            print(
                f"{classifier} {n_left_out}% left-out, {subject}, split {count} :",
                f"{conventional_result['accuracy']:.2f} | {stacked_result['accuracy']:.2f} / {stacked_result['dummy_accuracy']:.2f}, {fitted_classifiers_}",
            )

        count += 1

    results = pd.DataFrame(results)
    results.to_pickle(
        os.path.join(results_dir, f"results_clf_{classifier}_{subject}.pkl")
    )
    return results


def generate_sub_clf_combinations(subjects, classifiers):
    for subject_i, subject in enumerate(subjects):
        for clf in classifiers:
            yield subject, subject_i, clf


# vary number of samples left out for testing
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

df = pd.concat(all_results)
df["setting_classifier"] = df["setting"] + "_" + df["classifier"]
sns.pointplot(data=df, x="train_size", y="accuracy", hue="setting_classifier")
plt.axhline(y=df["dummy_accuracy"].mean(), color="k", linestyle="--")
plt.ylabel("Accuracy")
plt.xlabel("Training size")
plt.savefig(os.path.join(results_dir, f"results_{start_time}.png"))
plt.close()

sns.boxplot(data=df, x="train_size", y="accuracy", hue="setting_classifier")
plt.axhline(y=df["dummy_accuracy"].mean(), color="k", linestyle="--")
plt.ylabel("Accuracy")
plt.xlabel("Training size")
plt.savefig(os.path.join(results_dir, f"box_results_{start_time}.png"))
plt.close()
