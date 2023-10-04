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
def classify(clf, dummy_clf, train, test, X, Y, setting, count):
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

    print(
        f" {setting} split {count}",
        f"{accuracy:.2f} / {dummy_accuracy:.2f}",
    )

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
for n_left_out in tqdm(range(10, 60, 10)):
    print(f"Leaving {n_left_out}% of trials out")
    cv = StratifiedShuffleSplit(test_size=n_left_out / 100, random_state=0)
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
plt.axhline(y=results["dummy_accuracy"].mean(), color="k", linestyle="--")
plt.ylabel("Accuracy")
plt.xlabel("Training size")
plt.title("Within run")
plt.savefig(f"results_{time.strftime('%Y%m%d-%H%M%S')}.png")
plt.close()