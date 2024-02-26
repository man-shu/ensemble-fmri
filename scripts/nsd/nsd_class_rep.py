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
### get counts for each supercat
for i in tqdm(range(img_ids.shape[0]), desc="i", position=0):
    unique, counts = np.unique(supercat[i, :], return_counts=True)
    cat_counts = dict(zip(unique, counts))

total_trials_kept = 22500
### keep n first images for each sub
supercat = supercat[:, :total_trials_kept]
img_ids = img_ids[:, :total_trials_kept]

### get counts for each supercat
for i in tqdm(range(img_ids.shape[0]), desc="i", position=0):
    unique, counts = np.unique(supercat[i, :], return_counts=True)
    cat_counts = dict(zip(unique, counts))
    print(f"sub-{i}", cat_counts)

### see if images are repeated within a subject
for i in tqdm(range(img_ids.shape[0]), desc="i", position=0):
    unique, counts = np.unique(img_ids[i, :], return_counts=True)
    img_counts = dict(zip(unique, counts))
    repeated_images = 0
    for k, v in img_counts.items():
        if v > 1:
            cat = all_data[all_data["id"] == k]["supercat"].values[0]
            repeated_images = repeated_images + v - 1
            # print(f"subject {i} has {v} of image {k} of category {cat}")
    print(f"trials to remove for sub {i}:", repeated_images)

# reps in first 500 trials is 98
# reps in first 22500 trials is 13291
# same for each subject

### remove repeated images
supercat_no_reps = np.zeros(
    (supercat.shape[0], supercat.shape[1] - repeated_images), dtype="object"
)
img_ids_no_reps = np.zeros(
    (img_ids.shape[0], img_ids.shape[1] - repeated_images), dtype="int"
)
for sub in tqdm(range(img_ids.shape[0]), desc="subjects", position=0):
    (
        unique,
        idx,
        counts,
    ) = np.unique(img_ids[i, :], return_counts=True, return_index=True)
    supercat_no_reps[sub][:] = supercat[sub][idx]
    img_ids_no_reps[sub][:] = img_ids[sub][idx]

### check class representation after removing repeated images
for i in tqdm(range(img_ids.shape[0]), desc="i", position=0):
    unique, counts = np.unique(supercat_no_reps[i, :], return_counts=True)
    cat_counts = dict(zip(unique, counts))
    print(f"sub-{i}", cat_counts)

# keeping only four classes (animal, furniture, vehicle, person)
# in first 22500 trials, each class has at least 1212 trials (minimum for furniture, max for person 3003)
# in first 500 trials, each class has at least 45 trials (minimum for furniture, max for person not sure how many)
### so keep only first 1212 trials for each class
n_trials = 1212
classes = ["animal", "furniture", "vehicle", "person"]

ROOT = "/storage/store2/work/haggarwa/retreat_2023/data/"
OUT_DIR = os.path.join(ROOT, "nsd")
three_mm = os.path.join(OUT_DIR, f"3mm_{total_trials_kept}trials")
unselected_3mm = os.path.join(ROOT, "natural_scenes", "3mm_corrected")
three_mm_select = os.path.join(
    OUT_DIR, f"3mm_selected{n_trials}from{total_trials_kept}trials"
)
masks_dir = os.path.join(three_mm_select, "masks")
os.makedirs(masks_dir, exist_ok=True)

supercat_select = np.zeros(
    (supercat.shape[0], n_trials * len(classes)), dtype="object"
)
for sub in tqdm(range(img_ids.shape[0]), desc="subjects", position=0):
    idx = []
    for cat in ["animal", "furniture", "vehicle", "person"]:
        cat_idx = np.argwhere(supercat_no_reps[sub][:] == cat)[
            :n_trials
        ].squeeze()
        idx.extend(cat_idx)

    idx = np.sort(idx)
    print(idx.shape)
    splits = []
    import math

    for i in range(1, math.ceil(idx.max() / 750) + 1):
        split = np.delete(
            idx, np.where((idx < (i - 1) * 750) | (idx > i * 750 - 1))
        )
        split = split - ((i - 1) * 750)
        with open(
            os.path.join(masks_dir, f"subj0{sub+1}_session{i:02}_mask.npy"),
            "wb",
        ) as f:
            np.save(f, split)
        f.close()

    supercat_select[sub][:] = supercat_no_reps[sub][idx]

    np.savetxt(
        os.path.join(three_mm_select, f"subj0{sub+1}_labels.csv"),
        supercat_select[sub][:].T.astype(str),
        fmt="%s",
    )

    # ### index volumes
    # nifti = os.path.join(three_mm, f"subj0{sub+1}.nii.gz")
    # nifti = image.index_img(nifti, idx)
    # print(f"new shape of {nifti}: {nifti.shape}")
    # nifti.to_filename(os.path.join(three_mm_select, f"subj0{sub+1}.nii.gz"))


# check sizes of images for each subject, each session
for sub in tqdm(range(img_ids.shape[0]), desc="subjects", position=0):
    sessions = glob(os.path.join(unselected_3mm, f"subj0{sub+1}", "betas*"))
    for session in tqdm(sessions, desc="sessions", position=1):
        try:
            nifti = image.load_img(session)
        except Exception as e:
            print(e)
            print(f"error loading {session}")
            continue
        print(f"\nshape of subj0{sub+1} {session}: {nifti.shape}")

# keep specific images for each subject, each session
for sub in tqdm(range(img_ids.shape[0]), desc="subjects", position=0):
    sessions = glob(os.path.join(unselected_3mm, f"subj0{sub+1}", "betas*"))
    niftis = []
    # mask_lens = []
    for session in tqdm(sessions, desc="sessions", position=1):
        try:
            mask = np.load(
                os.path.join(
                    masks_dir,
                    f"subj0{sub+1}_{os.path.basename(session).split('.')[0].split('_')[1]}_mask.npy",
                )
            )
        except Exception as e:
            # print(e)
            print(
                f"no mask for {os.path.basename(session).split('.')[0].split('_')[1]}"
            )
            continue

        # print(mask.shape)
        # mask_lens.append(mask.shape[0])

        print(f"\nindexing {session}...")
        nifti = image.index_img(session, mask)
        print(f"\nnew shape of {nifti}: {nifti.get_fdata().shape}")
        niftis.append(nifti)
    # print(f"subj0{sub+1} mask len = {np.array(mask_lens).sum()}")
    out_file = os.path.join(
        three_mm_select,
        f"subj0{sub+1}.nii.gz",
    )
    subject_nifti = image.concat_imgs(niftis)
    print(f"\nnew shape of {subject_nifti}: {subject_nifti.get_fdata().shape}")
    print(f"\nsaving {out_file}...")
    subject_nifti.to_filename(out_file)
