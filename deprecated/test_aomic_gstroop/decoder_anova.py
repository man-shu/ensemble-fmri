import pandas as pd
from nilearn import maskers, image
import numpy as np
import os
from sklearn.base import clone
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
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
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.pipeline import Pipeline


DATA_ROOT = "/storage/store2/work/haggarwa/retreat_2023/test_aomic_gstroop/"
OUT_ROOT = "/storage/store2/work/haggarwa/retreat_2023/test_aomic_gstroop"
dataset = "aomic_gstroop"

# input data root path
data_dir = os.path.join(DATA_ROOT, dataset)
data_resolution = "3mm"  # or 1_5mm
nifti_dir = os.path.join(data_dir, data_resolution)

# create fake, empty Bunch object
atlas = Bunch()
atlas.name = "wholebrain"

## not doing difumo
# get difumo atlas
# atlas = datasets.fetch_atlas_difumo(
#     dimension=1024,
#     resolution_mm=3,
#     data_dir=DATA_ROOT,
#     legacy_format=False,
# )
# atlas["name"] = "difumo"

# output results path
start_time = time.strftime("%Y%m%d-%H%M%S")
results_dir = f"{dataset}_{atlas.name}_results_{start_time}"
results_dir = os.path.join(OUT_ROOT, results_dir)
os.makedirs(results_dir, exist_ok=True)

# get file names
imgs = glob(os.path.join(nifti_dir, "*.nii.gz"))
subjects = [os.path.basename(img).split(".")[0] for img in imgs]

# empty dictionary to store data
data = dict(responses=[], conditions=[], runs=[], subjects=[])

# store masked images
masked_dir = os.path.join(data_dir, atlas.name)
os.makedirs(masked_dir, exist_ok=True)
mask = load_mni152_gm_mask(resolution=3)

### combine two subjects data
combined_responses = []
combined_conditions = []
combined_runs = []
combined_subjects = []
for img, subject in zip(imgs, subjects):
    masked_file = os.path.join(masked_dir, f"{subject}.npy")
    # mask the image
    if os.path.exists(masked_file):
        masked = np.load(masked_file)
    else:
        if atlas.name == "wholebrain":
            masker = maskers.NiftiMasker(
                mask_img=mask,
                verbose=11,
            )
        else:
            masker = maskers.NiftiMapsMasker(
                maps_img=atlas["maps"],
                verbose=11,
            )
        masked = masker.fit_transform(img)
        np.save(masked_file, masked)

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
    num_trials = masked.shape[0]
    subs = np.repeat(subject, num_trials)

    combined_responses.append(masked)
    combined_conditions.append(conditions)
    combined_runs.append(runs)
    combined_subjects.append(subs)

# store data
data["responses"] = np.concatenate(combined_responses)
data["conditions"] = np.concatenate(combined_conditions)
data["runs"] = np.concatenate(combined_runs)
data["subjects"] = np.concatenate(combined_subjects)

N_JOBS = 50

cv = StratifiedShuffleSplit(n_splits=50, test_size=0.20, random_state=42)
# evaluation metrics
scoring = [
    "accuracy",
    "balanced_accuracy",
    "roc_auc_ovr_weighted",
    "roc_auc_ovr",
    "roc_auc_ovo_weighted",
    "roc_auc_ovo",
]

feature_selection = SelectPercentile(f_classif, percentile=10)
anova_svc = Pipeline(
    [("anova", feature_selection), ("svc", LinearSVC(dual=True))]
)
# try with LDA
# anova_lda = Pipeline(
#     [
#         ("anova", feature_selection),
#         ("lda", LinearDiscriminantAnalysis()),
#     ]
# )
anova_logistic = Pipeline(
    [("anova", feature_selection), ("logistic", LogisticRegression())]
)
anova_dummy = Pipeline(
    [("anova", feature_selection), ("dummy", DummyClassifier())]
)

fitted_svc = cross_validate(
    anova_svc,
    X=data["responses"],
    y=data["conditions"],
    n_jobs=50,
    cv=cv,
    scoring=scoring,
    verbose=11,
    return_train_score=True,
    return_estimator=True,
)

fitted_dummy = cross_validate(
    anova_dummy,
    X=data["responses"],
    y=data["conditions"],
    n_jobs=50,
    cv=cv,
    scoring=scoring,
    verbose=11,
    return_train_score=True,
    return_estimator=True,
)

print(
    f"{np.mean(fitted_svc['test_balanced_accuracy'])} | {np.mean(fitted_dummy['test_balanced_accuracy'])}"
)

# retrieve the pipeline fitted on the first cross-validation fold and its SVC
# coefficients
first_pipeline = fitted_svc["estimator"][0]
svc_coef = first_pipeline.named_steps["svc"].coef_
print(
    "After feature selection, "
    f"the SVC is trained only on {svc_coef.shape[1]} features"
)

# We invert the feature selection step to put these coefs in the right 2D place
full_coef = first_pipeline.named_steps["anova"].inverse_transform(svc_coef)

print(
    "After inverting feature selection, "
    f"we have {full_coef.shape[1]} features back"
)

# We apply the inverse of masking on these to make a 4D image that we can plot
weight_img = masker.inverse_transform(full_coef)
plot_stat_map(
    weight_img,
    title="Anova+SVC weights",
    display_mode="z",
    output_file="weights_anova.png",
)
