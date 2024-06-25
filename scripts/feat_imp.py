"""
Compute feature importance using the BlockBasedImportance method.
We only compute the feature importance for the stacked classifier, using
DiFuMo features and the Random Forest classiifer as the final classifier
(which is the only one we change, the first classifier is still LinearSVC).
These scores correspond to DiFuMo features but can later be projected to
the full-voxel feature space.

Note that this computation is independent of the other decoding
experiments. The BlockBasedImportance estimator internally fits the
decoding object and then computes the importance scores.
"""
import numpy as np
import os
from joblib import Parallel, delayed
import time
from glob import glob
import importlib.util
import sys
from tqdm import tqdm
from nilearn import datasets

if len(sys.argv) != 5:
    raise ValueError(
        "Please provide the following arguments in that order: ",
        "path to data, path to output, N parallel jobs.\n",
        "For example: ",
        "python scripts/feat_imp.py data results 20\n",
    )
else:
    DATA_ROOT = sys.argv[1]
    OUT_ROOT = sys.argv[2]
    N_JOBS = sys.argv[3]

# load local utility functions
spec = importlib.util.spec_from_file_location(
    "utils",
    os.path.join(OUT_ROOT, "utils.py"),
)
utils = importlib.util.module_from_spec(spec)
sys.modules["utils"] = utils
spec.loader.exec_module(utils)

# datasets and classifiers to use
datas = [
    "neuromod",
    "forrest",
    "rsvp",
    "aomic_anticipation",
]
classifiers = ["RandomForest"]

for dataset in datas:
    # input data root path
    data_dir = os.path.join(DATA_ROOT, dataset)
    data_resolution = "3mm"  # or 1_5mm
    nifti_dir = os.path.join(data_dir, data_resolution)

    # get difumo atlas
    atlas = datasets.fetch_atlas_difumo(
        dimension=1024,
        resolution_mm=3,
        data_dir=DATA_ROOT,
        legacy_format=False,
    )
    atlas["name"] = "difumo"

    # output results path
    results_dir = f"{dataset}_{atlas["name"]}_featimp"
    results_dir = os.path.join(OUT_ROOT, results_dir)
    os.makedirs(results_dir, exist_ok=True)

    # get file names
    imgs = glob(os.path.join(nifti_dir, "*.nii.gz"))
    subjects = [os.path.basename(img).split(".")[0] for img in imgs]

    print(f"\nParcellating {dataset}...")
    data = Parallel(n_jobs=N_JOBS, verbose=11, backend="multiprocessing")(
        delayed(utils.parcellate)(
            imgs[i],
            subject,
            atlas,
            data_dir=data_dir,
            nifti_dir=nifti_dir,
        )
        for i, subject in enumerate(subjects)
    )

    print(f"\nConcatenating all data for {dataset}...")
    data_ = dict(responses=[], conditions=[], runs=[], subjects=[])
    for entry in data:
        for key in ["responses", "conditions", "runs", "subjects"]:
            data_[key].extend(entry[key])
    for key in ["responses", "conditions", "runs", "subjects"]:
        data_[key] = np.array(data_[key])
    data = data_.copy()
    del data_

    print(f"\nPretraining dummy classifiers on {dataset}...")
    dummy_fitted_classifiers = Parallel(
        n_jobs=N_JOBS, verbose=11, backend="multiprocessing"
    )(
        delayed(utils.pretrain)(
            subject=subject,
            data=data,
            dummy=True,
            data_dir=data_dir,
            atlas=atlas,
        )
        for subject in subjects
    )

    print(f"\nPretraining linear classifiers on {dataset}...")
    fitted_classifiers = Parallel(
        n_jobs=N_JOBS, verbose=11, backend="multiprocessing"
    )(
        delayed(utils.pretrain)(
            subject=subject,
            data=data,
            dummy=False,
            data_dir=data_dir,
            atlas=atlas,
        )
        for subject in subjects
    )

    all_results = []
    for subject, subject_i, clf in tqdm(
        utils.generate_sub_clf_combinations(subjects, classifiers),
        total=len(subjects) * len(classifiers),
        desc=f"Running {dataset}",
    ):
        print(subject, subject_i)
        result = utils.feature_importance(
            subject,
            subject_i,
            data,
            clf,
            fitted_classifiers,
            dummy_fitted_classifiers,
            results_dir,
            dataset,
            n_jobs=N_JOBS,
        )
        all_results.append(result)
