"""This script curates the NSD dataset to only keep first 22500 samples 
per subject, remove samples where coco images with unknown categories 
were shown and finally remove some 'person' category samples to 
have equal number of samples per subject."""

import scipy.io
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from nilearn import image
from glob import glob

NSD_DIR = "/storage/store3/data/natural_scenes/"
exp_design = scipy.io.loadmat(
    os.path.join(NSD_DIR, "info", "nsd_expdesign.mat")
)

img_ids = exp_design["subjectim"][
    :, exp_design["masterordering"] - 1
].squeeze()

all_data = pd.read_pickle(
    os.path.join(NSD_DIR, "info", "coco_annotations", "coco_categories.pkl")
)

### create supercat matrix of shape (8 subjects, 30000 trials)
print("creating supercat label matrix...")
supercat = np.zeros(img_ids.shape, dtype="object")
for sub in tqdm(range(img_ids.shape[0]), desc="subjects", position=0):
    for trial in tqdm(range(img_ids.shape[1]), desc="trials", position=1):
        img_id = img_ids[sub, trial]
        supercat[sub, trial] = all_data[all_data["id"] == img_id][
            "supercat"
        ].values[0]
supercat_labels_file = os.path.join(
    NSD_DIR, "curated_3mm", "supercat_labels.npy"
)
np.save(supercat_labels_file, supercat)
print(f"saving supercat label matrix as {supercat_labels_file}...")

### create run number matrices
print("creating run label matrix...")
n_sessions = 40
n_runs_per_session = 12
n_trials_odd_runs = 63
n_trials_even_runs = 62
run_numbers = np.zeros(img_ids.shape, dtype="int")
for i in tqdm(range(img_ids.shape[0]), desc="i", position=0):
    session_runs = []
    run = 1
    for session in range(n_sessions):
        while run < (n_sessions * n_runs_per_session) + 1:
            if run % 2 == 0:
                numbers = np.repeat(run, n_trials_even_runs)
            else:
                numbers = np.repeat(run, n_trials_odd_runs)
            run += 1
            session_runs.extend(numbers)
    run_numbers[i, :] = session_runs
run_numbers_file = os.path.join(NSD_DIR, "curated_3mm", "runs.npy")
np.save(os.path.join(NSD_DIR, "curated_3mm", "runs.npy"), run_numbers)
print(f"saving run label matrix as {run_numbers_file}...")

print(
    "creating mask for trials not done by all subjects and trials with unknown categories..."
)
### start the process to remove unknowns and trials not done by all subjects
# keep only first 22500 images per subject, because the rest are not done by all subjects
# so set the rest to unknown to remove at a later stage
supercat[:, 22500:] = "unknown"
# check class representation and get number of unknowns per subject
unknown_counts = []
for i in tqdm(range(img_ids.shape[0]), desc="i", position=0):
    unique, counts = np.unique(supercat[i, :], return_counts=True)
    cat_counts = dict(zip(unique, counts))
    unknown_counts.append(cat_counts["unknown"])
# set some 'person' images to 'unknown', such that we have equal unknowns per subject
max_unknown = np.max(unknown_counts)
for i in range(img_ids.shape[0]):
    unknown_needed = max_unknown - unknown_counts[i]
    all_persons = np.where(supercat[i, :] == "person")[0]
    for j in range(unknown_needed):
        supercat[i, all_persons[j]] = "unknown"
# get a mask where supercat is unknown
unknown_mask = supercat == "unknown"
# save mask
unknown_mask_file = os.path.join(NSD_DIR, "curated_3mm", "unknown_mask.npy")
np.save(os.path.join(NSD_DIR, "curated_3mm", "unknown_mask.npy"), unknown_mask)
print(f"saving unknowns mask matrix as {unknown_mask_file}...")

print("applying the mask to remove nifti volumes, labels and runs...")
### start the process to remove unknowns and trials not done by all subjects
# load mask
unknown_mask = np.load(
    os.path.join(NSD_DIR, "curated_3mm", "unknown_mask.npy"), allow_pickle=True
)
# load supercat labels
supercat = np.load(
    os.path.join(NSD_DIR, "curated_3mm", "supercat_labels.npy"),
    allow_pickle=True,
)
# load run numbers
run_numbers = np.load(
    os.path.join(NSD_DIR, "curated_3mm", "runs.npy"), allow_pickle=True
)
### concatenate and remove nifti volumes corresponding to unknown supercats
three_mm = os.path.join(NSD_DIR, "3mm")
subs = os.listdir(three_mm)
out_subs = [f"sub-{i+1:02d}" for i in range(len(subs))]
for i, sub in tqdm(
    enumerate(subs), desc="subjects", position=0, total=len(subs)
):
    sub_dir = os.path.join(three_mm, sub)
    niftis = glob(os.path.join(sub_dir, "betas*.nii.gz"))
    print("concatenating...")
    concatenated = image.concat_imgs(niftis)
    print(concatenated.shape)
    print("masking...")
    try:
        concatenated = image.index_img(
            concatenated, np.invert(unknown_mask[i])
        )
    except IndexError:
        print(f"{out_subs[i]} only did {concatenated.shape[3]} trials")
        print(f"reshaping mask to {concatenated.shape[3]} trials")
        concatenated = image.index_img(
            concatenated, np.invert(unknown_mask[i][: concatenated.shape[3]])
        )
    print(concatenated.shape)
    print("saving...")
    concatenated.to_filename(
        os.path.join(NSD_DIR, "curated_3mm", f"{out_subs[i]}.nii.gz")
    )
    print("mask and save labels...")
    supercat_ = supercat[i, np.invert(unknown_mask[i])]
    np.save(
        os.path.join(NSD_DIR, "curated_3mm", f"{out_subs[i]}_labels.npy"),
        supercat_,
    )
    print("mask and save runs...")
    run_numbers_ = run_numbers[i, np.invert(unknown_mask[i])]
    np.save(
        os.path.join(NSD_DIR, "curated_3mm", f"{out_subs[i]}_runs.npy"),
        supercat_,
    )
