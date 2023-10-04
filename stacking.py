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

# trial conditions
conditions = pd.read_csv(os.path.join(data_dir, "labels.csv"), header=None)
conditions = conditions[0].values
# run labels
runs = pd.read_csv(os.path.join(data_dir, "runs.csv"), header=None)
runs = runs[0].values
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

results = []
for subject in subjects:
    unique_runs = np.unique(data["runs"])
    fitted_classifiers = []
    dummy_fitted_classifiers = []
    for run in tqdm(unique_runs):
        clf = LinearSVC(dual="auto")
        dummy_clf = DummyClassifier(strategy="most_frequent")
        # select data for current run
        sub_run_mask = np.where(
            (data["subjects"] == subject) & (data["runs"] == run)
        )[0]
        X = data["responses"][sub_run_mask]
        y = data["conditions"][sub_run_mask]
        # fit classifier
        clf.fit(X, y)
        dummy_clf.fit(X, y)
        # save classifier
        fitted_classifiers.append((f"run-0{run}", clf))
        dummy_fitted_classifiers.append((f"run-0{run}", dummy_clf))

    # Stack classifiers
    for run in tqdm(unique_runs):
        print(f"Subject {subject}, Run 0{run}")
        cv = StratifiedShuffleSplit(n_splits=30, test_size=0.2, random_state=0)
        # select data for current sub, run
        sub_run_mask = np.where(
            (data["subjects"] == subject) & (data["runs"] == run)
        )[0]
        X = data["responses"][sub_run_mask]
        y = data["conditions"][sub_run_mask]
        print("Stacking classifiers...")
        # remove current run from fitted classifiers
        fitted_classifiers_ = fitted_classifiers.copy()
        fitted_classifiers_.pop(run)
        dummy_fitted_classifiers_ = dummy_fitted_classifiers.copy()
        dummy_fitted_classifiers_.pop(run)
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
        count = 0
        for train, test in cv.split(X, y):
            stacked_clf.fit(X[train], y[train])
            dummy_stacked_clf.fit(X[train], y[train])
            prediction = stacked_clf.predict(X[test])
            dummy_prediction = dummy_stacked_clf.predict(X[test])
            accuracy = accuracy_score(y[test], prediction)
            dummy_accuracy = accuracy_score(y[test], dummy_prediction)
            print(
                f" split {count}",
                f"{accuracy:.2f} / {dummy_accuracy:.2f}",
            )
            result = {}
            result["run"] = run
            result["accuracy"] = accuracy
            result["dummy_accuracy"] = dummy_accuracy
            result["subject"] = subject
            result["true"] = y[test]
            result["predicted"] = prediction
            result["dummy_predicted"] = dummy_prediction
            results.append(result)
            count += 1

results = pd.DataFrame(results)
results.to_pickle(
    f"within_sub_stacked_results_{time.strftime('%Y%m%d-%H%M%S')}.pkl"
)

print("Plotting within subject results...")

sns.barplot(
    data=results,
    x="subject",
    y="accuracy",
    palette=sns.color_palette(),
)
sns.barplot(
    data=results,
    x="subject",
    y="dummy_accuracy",
    palette=sns.color_palette("pastel"),
)
plt.axhline(y=results["accuracy"].mean(), color="k", linestyle="--")
plt.text(
    x=-1.3,
    y=results["accuracy"].mean(),
    s=f"{round(results['accuracy'].mean(), 2)}",
    color="k",
)
plt.text(
    x=0.34,
    y=results["accuracy"].mean() + 0.01,
    s="Mean Accuracy",
    color="k",
)
plt.ylabel("Accuracy")
plt.xlabel("Subject")
plt.xticks(rotation=30)
plt.title("Within subject")
plt.savefig(f"within_sub_stacked_results_{time.strftime('%Y%m%d-%H%M%S')}.png")
plt.close()


#### across subjects ####
fitted_classifiers = []
dummy_fitted_classifiers = []
for subject in tqdm(subjects):
    clf = LinearSVC(dual="auto")
    dummy_clf = DummyClassifier(strategy="most_frequent")
    # select data for current subject
    sub_mask = np.where(data["subjects"] == subject)[0]
    X = data["responses"][sub_mask]
    print(X.shape)
    y = data["conditions"][sub_mask]
    print(y.shape)
    # fit classifier
    clf.fit(X, y)
    dummy_clf.fit(X, y)
    # save classifier
    fitted_classifiers.append((f"{subject}", clf))
    dummy_fitted_classifiers.append((f"{subject}", dummy_clf))

# Stack classifiers
results = []
print("Stacking classifiers...")
for sub_i, subject in tqdm(enumerate(subjects)):
    print(f"Subject {subject}")
    cv = LeaveOneGroupOut()
    # select data for current subject
    sub_mask = np.where(data["subjects"] == subject)[0]
    X = data["responses"][sub_mask]
    y = data["conditions"][sub_mask]
    groups = data["runs"][sub_mask]
    # remove current subject from fitted classifiers
    fitted_classifiers_ = fitted_classifiers.copy()
    fitted_classifiers_.pop(sub_i)
    dummy_fitted_classifiers_ = dummy_fitted_classifiers.copy()
    dummy_fitted_classifiers_.pop(sub_i)
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
    count = 0

    for train, test in cv.split(X, y, groups=groups):
        stacked_clf.fit(X[train], y[train])
        dummy_stacked_clf.fit(X[train], y[train])
        prediction = stacked_clf.predict(X[test])
        dummy_prediction = dummy_stacked_clf.predict(X[test])
        accuracy = accuracy_score(y[test], prediction)
        dummy_accuracy = accuracy_score(y[test], dummy_prediction)
        print(
            f" split {count}",
            f"{accuracy:.2f} / {dummy_accuracy:.2f}",
        )
        result = {}
        result["accuracy"] = accuracy
        result["dummy_accuracy"] = dummy_accuracy
        result["subject"] = subject
        result["true"] = y[test]
        result["predicted"] = prediction
        result["dummy_predicted"] = dummy_prediction
        results.append(result)
        count += 1

results = pd.DataFrame(results)
results.to_pickle(
    f"across_sub_stacked_results_{time.strftime('%Y%m%d-%H%M%S')}.pkl"
)

# plot results
print("Plotting across subjects results...")
sns.barplot(
    data=results,
    x="subject",
    y="accuracy",
    palette=sns.color_palette(),
)
sns.barplot(
    data=results,
    x="subject",
    y="dummy_accuracy",
    palette=sns.color_palette("pastel"),
)
plt.axhline(y=results["accuracy"].mean(), color="k", linestyle="--")
plt.text(
    x=-1.3,
    y=results["accuracy"].mean(),
    s=f"{round(results['accuracy'].mean(), 2)}",
    color="k",
)
plt.text(
    x=0.34,
    y=results["accuracy"].mean() + 0.01,
    s="Mean Accuracy",
    color="k",
)
plt.ylabel("Accuracy")
plt.xlabel("Subject")
plt.xticks(rotation=30)
plt.title("Across Subjects")
plt.savefig(f"across_sub_stacked_results_{time.strftime('%Y%m%d-%H%M%S')}.png")
plt.close()

##### pretrain across subjects
##### cv within subject
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

results = []
for sub_i, subject in tqdm(enumerate(subjects)):
    unique_runs = np.unique(data["runs"])
    # remove current run from fitted classifiers
    fitted_classifiers_ = fitted_classifiers.copy()
    fitted_classifiers_.pop(sub_i)
    dummy_fitted_classifiers_ = dummy_fitted_classifiers.copy()
    dummy_fitted_classifiers_.pop(sub_i)
    print("Stacking classifiers...")
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
    # Stack classifiers
    for run in unique_runs:
        print(f"Subject {subject}, Run 0{run}")
        cv = StratifiedShuffleSplit(n_splits=30, test_size=0.2, random_state=0)
        # select data for current sub, run
        sub_run_mask = np.where(
            (data["subjects"] == subject) & (data["runs"] == run)
        )[0]
        X = data["responses"][sub_run_mask]
        y = data["conditions"][sub_run_mask]
        count = 0
        for train, test in cv.split(X, y):
            stacked_clf.fit(X[train], y[train])
            dummy_stacked_clf.fit(X[train], y[train])
            prediction = stacked_clf.predict(X[test])
            dummy_prediction = dummy_stacked_clf.predict(X[test])
            accuracy = accuracy_score(y[test], prediction)
            dummy_accuracy = accuracy_score(y[test], dummy_prediction)
            print(
                f" split {count}",
                f"{accuracy:.2f} / {dummy_accuracy:.2f}",
            )
            result = {}
            result["run"] = run
            result["accuracy"] = accuracy
            result["dummy_accuracy"] = dummy_accuracy
            result["subject"] = subject
            result["true"] = y[test]
            result["predicted"] = prediction
            result["dummy_predicted"] = dummy_prediction
            results.append(result)
            count += 1

results = pd.DataFrame(results)
results.to_pickle(
    f"across_pretrain_within_sub_stacked_results_{time.strftime('%Y%m%d-%H%M%S')}.pkl"
)

print("Plotting results...")

sns.barplot(
    data=results,
    x="subject",
    y="accuracy",
    palette=sns.color_palette(),
)
sns.barplot(
    data=results,
    x="subject",
    y="dummy_accuracy",
    palette=sns.color_palette("pastel"),
)
plt.axhline(y=results["accuracy"].mean(), color="k", linestyle="--")
plt.text(
    x=-1.3,
    y=results["accuracy"].mean(),
    s=f"{round(results['accuracy'].mean(), 2)}",
    color="k",
)
plt.text(
    x=0.34,
    y=results["accuracy"].mean() + 0.01,
    s="Mean Accuracy",
    color="k",
)
plt.ylabel("Accuracy")
plt.xlabel("Subject")
plt.xticks(rotation=30)
plt.title("Within subject")
plt.savefig(
    f"across_pretrain_within_sub_stacked_results_{time.strftime('%Y%m%d-%H%M%S')}.png"
)
plt.close()
