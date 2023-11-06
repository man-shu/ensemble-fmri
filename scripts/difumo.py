import pandas as pd
from nilearn import image, datasets, maskers
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
DATA_ROOT = "/storage/store2/work/haggarwa/retreat_2023/data/"
data_dir = os.path.join(DATA_ROOT, "rsvp_trial")
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
# parcellate data with atlas
# get difumo atlas
atlas = datasets.fetch_atlas_difumo(
    dimension=1024, resolution_mm=3, data_dir=DATA_ROOT
)
masker = maskers.MultiNiftiMapsMasker(
    maps_img=atlas.maps, memory=DATA_ROOT, verbose=11, n_jobs=20
)
# get list of images
imgs = [os.path.join(nifti_dir, f"{subject}.nii.gz") for subject in subjects]
# create empty dictionary to store data
data = dict(responses=[], conditions=[], runs=[], subjects=[])
# parcellate data
print("Parcellating data...")
data["responses"] = masker.fit_transform(imgs)
