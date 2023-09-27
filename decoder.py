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
)
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import ibc_public.utils_data
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import time

data_dir = "rsvp_trial"
subjects_sessions = ibc_public.utils_data.get_subject_session("rsvp-language")
subjects = np.array(
    [subject for subject, session in subjects_sessions], dtype="object"
)
subjects = np.unique(subjects)
data = dict(responses=[], conditions=[], runs=[], subjects=[])
for subject in tqdm(subjects):
    try:
        response = image.load_img(
            os.path.join(data_dir, f"{subject}", f"{subject}.nii.gz")
        )
    except:
        print(f"{subject} not found")
        continue
    response = image.resample_img(response, target_affine=np.diag((3, 3, 3)))
    response.shape
    response = response.get_fdata()
    # reshape from (53, 63, 52, 360) to 2d array (360, 53*63*52)
    response = response.reshape(response.shape[-1], -1)
    num_trials = response.shape[0]
    subs = np.repeat(subject, num_trials)
    conditions = pd.read_csv(
        os.path.join(data_dir, "sub-01", "sub-01_labels.csv"), header=None
    )
    conditions = conditions[0].values
    runs = pd.read_csv(
        os.path.join(data_dir, "sub-01", "sub-01_runs.csv"), header=None
    )
    runs = runs[0].values

    data["responses"].append(response)
    data["conditions"].append(conditions)
    data["runs"].append(runs)
    data["subjects"].append(subs)

for dat in ["responses", "conditions", "runs", "subjects"]:
    data[dat] = np.concatenate(data[dat])
    print(data[dat].shape)


def classify(train, test, cv, X, Y, groups):
    result = {}
    clf = LinearSVC(dual="auto").fit(X[train], Y[train])
    dummy = DummyClassifier(strategy="most_frequent").fit(X[train], Y[train])
    prediction = clf.predict(X[test])
    dummy_prediction = dummy.predict(X[test])
    accuracy = accuracy_score(Y[test], prediction)
    dummy_accuracy = accuracy_score(Y[test], dummy_prediction)
    result["accuracy"] = accuracy
    result["dummy_accuracy"] = dummy_accuracy
    result["true"] = Y[test]
    result["predicted"] = prediction
    result["dummy_predicted"] = dummy_prediction
    result["train_group"] = np.unique(groups[train])
    result["test_group"] = np.unique(groups[test])[0]

    return result


cv = LeaveOneGroupOut()
X = data["responses"]
Y = data["conditions"]
groups = data["subjects"]
across_results = []
for train, test in tqdm(cv.split(X, Y, groups)):
    result = classify(train, test, cv, X, Y, groups)
    across_results.append(result)

across_results = pd.DataFrame(across_results)
across_results.to_pickle(
    f"across_results_{time.strftime('%Y%m%d-%H%M%S')}.pkl"
)
sns.barplot(
    data=across_results,
    x="test_group",
    y="accuracy",
    palette=sns.color_palette(),
)
sns.barplot(
    data=across_results,
    x="test_group",
    y="dummy_accuracy",
    palette=sns.color_palette("pastel"),
)
plt.axhline(y=across_results["accuracy"].mean(), color="k", linestyle="--")
plt.text(
    x=-1.3,
    y=across_results["accuracy"].mean(),
    s=f"{round(across_results['accuracy'].mean(), 2)}",
    color="k",
)
plt.text(
    x=0.34,
    y=across_results["accuracy"].mean() + 0.01,
    s="Mean Accuracy",
    color="k",
)
plt.ylabel("Accuracy")
plt.xlabel("Test Subject")
plt.xticks(rotation=30)
plt.title("Across Subjects")
plt.savefig(f"across_results_{time.strftime('%Y%m%d-%H%M%S')}.png")
# plt.savefig("results.png")
# del across_results
plt.close()

within_results = []
for subject in tqdm(subjects):
    sub_index = np.where(data["subjects"] == subject)[0]
    X = data["responses"][sub_index]
    Y = data["conditions"][sub_index]
    groups = data["runs"][sub_index]
    for train, test in tqdm(cv.split(X, Y, groups)):
        result = classify(train, test, cv, X, Y, groups)
        result["subject"] = subject
        within_results.append(result)

within_results = pd.DataFrame(within_results)
within_results.to_pickle(
    f"within_results_{time.strftime('%Y%m%d-%H%M%S')}.pkl"
)
print(within_results)


sns.barplot(
    data=within_results,
    x="subject",
    y="dummy_accuracy",
    palette=sns.color_palette("pastel"),
)
sns.barplot(
    data=within_results,
    x="subject",
    y="accuracy",
    palette=sns.color_palette(),
)
plt.axhline(y=within_results["accuracy"].mean(), color="k", linestyle="--")
plt.text(
    x=-1.3,
    y=within_results["accuracy"].mean(),
    s=f"{round(within_results['accuracy'].mean(), 2)}",
    color="k",
)
plt.text(
    x=0.34,
    y=within_results["accuracy"].mean() + 0.01,
    s="Mean Accuracy",
    color="k",
)
plt.ylabel("Accuracy")
plt.xlabel("Subject")
plt.xticks(rotation=30)
plt.title("Within subject")
plt.savefig(f"within_results_{time.strftime('%Y%m%d-%H%M%S')}.png")
# plt.savefig("results.png")
# del within_results
plt.close()


mixed_results = []
data["sub_run"] = np.array(
    [f"{sub}_{run}" for sub, run in zip(data["subjects"], data["runs"])],
    dtype="object",
)
for subject in tqdm(subjects):
    for run in range(0, 6):
        keep_index = np.where(
            ~((data["subjects"] == subject) & (data["runs"] == run))
        )[0]
        X = data["responses"][keep_index]
        Y = data["conditions"][keep_index]
        groups = data["sub_run"][keep_index]
        for train, test in tqdm(cv.split(X, Y, groups)):
            result = classify(train, test, cv, X, Y, groups)
            result["subject"] = subject
            mixed_results.append(result)

mixed_results = pd.DataFrame(mixed_results)
mixed_results.to_pickle(f"mixed_results_{time.strftime('%Y%m%d-%H%M%S')}.pkl")
print(mixed_results)

sns.barplot(
    data=mixed_results,
    x="subject",
    y="dummy_accuracy",
    palette=sns.color_palette("pastel"),
)
sns.barplot(
    data=mixed_results,
    x="subject",
    y="accuracy",
    palette=sns.color_palette(),
)
plt.axhline(y=mixed_results["accuracy"].mean(), color="k", linestyle="--")
plt.text(
    x=-1.3,
    y=mixed_results["accuracy"].mean(),
    s=f"{mixed_results['accuracy'].mean()}",
    color="k",
)
plt.text(
    x=0.34,
    y=mixed_results["accuracy"].mean() + 0.01,
    s="Mean Accuracy",
    color="k",
)
plt.ylabel("Accuracy")
plt.xlabel("Subject")
plt.savefig(f"mixed_results_{time.strftime('%Y%m%d-%H%M%S')}.png")
# plt.savefig("results.png")
# del mixed_results
plt.close()
