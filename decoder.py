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

data_dir = "../ibc_data/rsvp_trial/3mm/"
subjects_sessions = ibc_public.utils_data.get_subject_session("rsvp-language")
subjects = np.array(
    [subject for subject, session in subjects_sessions], dtype="object"
)
subjects = np.unique(subjects)
data = dict(responses=[], conditions=[], runs=[], subjects=[])
for subject in subjects:
    try:
        response = image.load_img(os.path.join(data_dir, f"{subject}.nii.gz"))
    except:
        print(f"{subject} not found")
        continue
    response = response.get_fdata()
    # reshape from (53, 63, 52, 360) to 2d array (360, 53*63*52)
    response = response.reshape(response.shape[-1], -1)
    num_trials = response.shape[0]
    subs = np.repeat(subject, num_trials)
    conditions = pd.read_csv(
        os.path.join(data_dir, f"{subject}_labels.csv"), header=None
    )
    conditions = conditions[0].values
    runs = pd.read_csv(
        os.path.join(data_dir, f"{subject}_runs.csv"), header=None
    )
    runs = runs[0].values

    data["responses"].append(response)
    data["conditions"].append(conditions)
    data["runs"].append(runs)
    data["subjects"].append(subs)

for dat in ["responses", "conditions", "runs", "subjects"]:
    data[dat] = np.concatenate(data[dat])
    print(data[dat].shape)

cv = LeaveOneGroupOut()
X = data["responses"]
Y = data["conditions"]
groups = data["subjects"]
results = []
for train, test in tqdm(cv.split(X, Y, groups)):
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
    result["train_subjects"] = np.unique(groups[train])
    result["test_subjects"] = np.unique(groups[test])[0]

    results.append(result)

results = pd.DataFrame(results)
results.to_pickle("results.pkl")
print(results)
print(results["accuracy"].mean())
print(results["dummy_accuracy"].mean())

sns.barplot(
    data=results, x="test_subjects", y="accuracy", palette=sns.color_palette()
)
sns.barplot(
    data=results,
    x="test_subjects",
    y="dummy_accuracy",
    palette=sns.color_palette("pastel"),
)
plt.savefig("results.png")
