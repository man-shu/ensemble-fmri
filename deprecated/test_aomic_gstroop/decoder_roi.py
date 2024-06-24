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
from nilearn.datasets import load_mni152_gm_mask
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import time
from glob import glob
import importlib.util
import sys
from sklearn.utils import Bunch
from sklearn.model_selection import cross_validate
from nilearn import datasets
from nilearn import decoding
from nilearn.plotting import (
    plot_design_matrix,
    plot_contrast_matrix,
    plot_stat_map,
)

DATA_ROOT = "/storage/store2/work/haggarwa/retreat_2023/test_aomic_gstroop/"
OUT_ROOT = "/storage/store2/work/haggarwa/retreat_2023/test_aomic_gstroop"
dataset = "aomic_gstroop"

# input data root path
data_dir = os.path.join(DATA_ROOT, dataset)
data_resolution = "3mm"  # or 1_5mm
nifti_dir = os.path.join(data_dir, data_resolution)

# get file names
imgs = glob(os.path.join(nifti_dir, "*.nii.gz"))
subjects = [os.path.basename(img).split(".")[0] for img in imgs]

# empty dictionary to store data
data = dict(responses=[], conditions=[], runs=[], subjects=[])

# load roi, created from using roi neurovault.py
roi = image.load_img(os.path.join(OUT_ROOT, "roi_neurovault.nii.gz"))

### combine two subjects data
combined_responses = []
combined_conditions = []
combined_runs = []
combined_subjects = []
for img, subject in zip(imgs, subjects):
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
    num_trials = conditions.shape[0]
    subs = np.repeat(subject, num_trials)

    combined_conditions.append(conditions)
    combined_runs.append(runs)
    combined_subjects.append(subs)

data["responses"] = image.concat_imgs(imgs)
data["conditions"] = np.concatenate(combined_conditions)
data["runs"] = np.concatenate(combined_runs)
data["subjects"] = np.concatenate(combined_subjects)

N_JOBS = 50

cv = StratifiedShuffleSplit(n_splits=200, test_size=0.20, random_state=42)

svc = decoding.Decoder(
    estimator="svc",
    mask=roi,
    cv=cv,
    n_jobs=N_JOBS,
    verbose=11,
    screening_percentile=100,
    standardize="zscore_sample",
    # scoring="accuracy",
)
svc.fit(X=data["responses"], y=data["conditions"])

chance = decoding.Decoder(
    estimator="dummy_classifier",
    mask=roi,
    cv=cv,
    n_jobs=N_JOBS,
    verbose=11,
    screening_percentile=100,
    standardize="zscore_sample",
    # scoring="accuracy",
)
chance.fit(X=data["responses"], y=data["conditions"])

print(
    f"{np.mean(svc.cv_scores_['left'] + svc.cv_scores_['right'])} | {np.mean(chance.cv_scores_['left'] + chance.cv_scores_['right'])}"
)

plot_stat_map(
    svc.coef_img_["left"],
    title="SVM weights",
    output_file="weights_left.png",
    display_mode="z",
)
plot_stat_map(
    svc.coef_img_["right"],
    title="SVM weights",
    output_file="weights_right.png",
    display_mode="z",
)
