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


# classification function
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


# cross validation scheme
cv = LeaveOneGroupOut()

# across subjects classification
# Train on 12 (out of 13) subjects and test on the left-out subject
X = data["responses"]
Y = data["conditions"]
groups = data["subjects"]
across_results = []
print("Running across subjects classification...")
for train, test in tqdm(cv.split(X, Y, groups)):
    result = classify(train, test, cv, X, Y, groups)
    across_results.append(result)
# save results
print("Saving across subjects results...")
across_results = pd.DataFrame(across_results)
across_results.to_pickle(
    f"across_results_{time.strftime('%Y%m%d-%H%M%S')}.pkl"
)
# plot results
print("Plotting across subjects results...")
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
plt.close()

# within subject classification
# Train on 5 (out 6) runs and test on left-out run for each subject
within_results = []
print("Running within subject classification...")
for subject in tqdm(subjects):
    sub_index = np.where(data["subjects"] == subject)[0]
    X = data["responses"][sub_index]
    Y = data["conditions"][sub_index]
    groups = data["runs"][sub_index]
    for train, test in cv.split(X, Y, groups):
        result = classify(train, test, cv, X, Y, groups)
        result["subject"] = subject
        within_results.append(result)

print("Saving within subject results...")
within_results = pd.DataFrame(within_results)
within_results.to_pickle(
    f"within_results_{time.strftime('%Y%m%d-%H%M%S')}.pkl"
)
print("Plotting within subject results...")

sns.barplot(
    data=within_results,
    x="subject",
    y="accuracy",
    palette=sns.color_palette(),
)
sns.barplot(
    data=within_results,
    x="subject",
    y="dummy_accuracy",
    palette=sns.color_palette("pastel"),
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
plt.close()


def _plot_cv_indices(
    cv,
    X,
    y,
    subject,
    run,
    n_splits,
    out_dir,
    lw=10,
):
    """Create a sample plot for indices of a cross-validation object."""
    fig, ax = plt.subplots()
    cmap_data = plt.cm.tab20
    cmap_cv = plt.cm.coolwarm
    _, y = np.unique(y, return_inverse=True)
    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X, y)):
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
    # Formatting
    yticklabels = [*range(n_splits)] + ["class"]
    ax.set(
        yticks=np.arange(n_splits + 1) + 0.5,
        yticklabels=yticklabels,
        xlabel="Sample index",
        ylabel="CV iteration",
        ylim=[n_splits + 2.2, -0.2],
    )
    split_dir = os.path.join(out_dir, "test_train_splits")
    os.makedirs(split_dir, exist_ok=True)
    ax.set_title(f"Train/test splits for classifying run {run} of {subject}")
    plot_file = f"{subject}_run-{run}_cv_indices.png"
    plot_file = os.path.join(split_dir, plot_file)
    fig.savefig(plot_file, bbox_inches="tight")
    plt.close()


# within run classification
# Train on 80 percent of trials and test on left-out 20% of trials for each run
# cross validation scheme
cv = StratifiedShuffleSplit(n_splits=30, test_size=0.2, random_state=0)
print("Running within run classification...")
within_run_results_dir = "within_run_results_n30_random_state0"
os.makedirs(within_run_results_dir, exist_ok=True)
for subject in tqdm(subjects):
    within_run_results = []
    for run in range(0, 6):
        sub_run_mask = np.where(
            (data["subjects"] == subject) & (data["runs"] == run)
        )[0]
        X = data["responses"][sub_run_mask]
        Y = data["conditions"][sub_run_mask]
        groups = data["runs"][sub_run_mask]
        count = 1
        _plot_cv_indices(cv, X, Y, subject, run, 10, within_run_results_dir)
        for train, test in cv.split(X, Y, groups):
            result = classify(train, test, cv, X, Y, groups)
            result["subject"] = subject
            result["run"] = run
            within_run_results.append(result)
            print(f"split {count} of run {run} of {subject} complete")
            print(
                f"accuracy / chance: {result['accuracy']:.2f} / {result['dummy_accuracy']:.2f}"
            )
            count += 1

    print("Saving within run results...")
    within_run_results = pd.DataFrame(within_run_results)
    within_run_results_pkl = os.path.join(
        within_run_results_dir,
        f"within_run_results_{subject}_{time.strftime('%Y%m%d-%H%M%S')}.pkl",
    )
    within_run_results.to_pickle(within_run_results_pkl)
    print("Plotting within run results...")
    plt.close()
    sns.barplot(
        data=within_run_results,
        x="run",
        y="accuracy",
        palette=sns.color_palette(),
    )
    sns.barplot(
        data=within_run_results,
        x="run",
        y="dummy_accuracy",
        palette=sns.color_palette("pastel"),
    )
    plt.axhline(
        y=within_run_results["accuracy"].mean(), color="k", linestyle="--"
    )
    plt.text(
        x=-1.3,
        y=within_run_results["accuracy"].mean(),
        s=f"{round(within_run_results['accuracy'].mean(), 2)}",
        color="k",
    )
    plt.text(
        x=0.34,
        y=within_run_results["accuracy"].mean() + 0.01,
        s="Mean Accuracy",
        color="k",
    )
    plt.ylabel("Accuracy")
    plt.xlabel("Run")
    plt.title(f"Within run for {subject}")
    within_run_results_plt = os.path.join(
        within_run_results_dir,
        f"within_run_results_{subject}_{time.strftime('%Y%m%d-%H%M%S')}.png",
    )
    plt.savefig(within_run_results_plt, bbox_inches="tight")
    plt.close()


# mixed_results = []
# data["sub_run"] = np.array(
#     [f"{sub}_{run}" for sub, run in zip(data["subjects"], data["runs"])],
#     dtype="object",
# )
# for subject in tqdm(subjects):
#     for run in range(0, 6):
#         keep_index = np.where(
#             ~((data["subjects"] == subject) & (data["runs"] == run))
#         )[0]
#         X = data["responses"][keep_index]
#         Y = data["conditions"][keep_index]
#         groups = data["sub_run"][keep_index]
#         for train, test in tqdm(cv.split(X, Y, groups)):
#             result = classify(train, test, cv, X, Y, groups)
#             result["subject"] = subject
#             mixed_results.append(result)

# mixed_results = pd.DataFrame(mixed_results)
# mixed_results.to_pickle(f"mixed_results_{time.strftime('%Y%m%d-%H%M%S')}.pkl")
# print(mixed_results)

# sns.barplot(
#     data=mixed_results,
#     x="subject",
#     y="dummy_accuracy",
#     palette=sns.color_palette("pastel"),
# )
# sns.barplot(
#     data=mixed_results,
#     x="subject",
#     y="accuracy",
#     palette=sns.color_palette(),
# )
# plt.axhline(y=mixed_results["accuracy"].mean(), color="k", linestyle="--")
# plt.text(
#     x=-1.3,
#     y=mixed_results["accuracy"].mean(),
#     s=f"{mixed_results['accuracy'].mean()}",
#     color="k",
# )
# plt.text(
#     x=0.34,
#     y=mixed_results["accuracy"].mean() + 0.01,
#     s="Mean Accuracy",
#     color="k",
# )
# plt.ylabel("Accuracy")
# plt.xlabel("Subject")
# plt.savefig(f"mixed_results_{time.strftime('%Y%m%d-%H%M%S')}.png")
# # plt.savefig("results.png")
# # del mixed_results
# plt.close()
