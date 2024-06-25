import scipy.io
import os
import pandas as pd
from pycocotools.coco import COCO
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from nilearn import image
from glob import glob
from nilearn.datasets import load_mni152_gm_mask
from nilearn.maskers import NiftiMasker
from nilearn.masking import unmask

NSD_DIR = "/storage/store3/data/natural_scenes/"
ANOT_ROOT = "/storage/store/work/haggarwa/"
PROJ_ROOT = "/storage/store2/work/haggarwa/retreat_2023/"
hugging_face_annots = os.path.join(PROJ_ROOT, "COCO_73k_annots_curated.npy")
hugging_face_annots = np.load(hugging_face_annots)

exp_design = scipy.io.loadmat(os.path.join(NSD_DIR, "nsd_expdesign.mat"))
image_data = pd.read_pickle(os.path.join(NSD_DIR, "nsd_stim_info_merged.pkl"))


img_ids = exp_design["subjectim"][
    :, exp_design["masterordering"] - 1
].squeeze()

merged_file = os.path.join(
    ANOT_ROOT, "Merge_COCO_FILES", "instances_train_val_2017.json"
)
coco = COCO(merged_file)

flat_img_ids = np.unique(img_ids.flatten())


def look_up_cats(img_ids, coco):
    all_data = []
    for img_id in tqdm(img_ids, desc="looking up cats"):
        data = {}
        data["id"] = img_id
        data["supercats"] = []
        data["cats"] = []
        # print("looking for image", img_id)
        for cat_id, imgs in coco.__dict__["catToImgs"].items():
            if img_id in imgs:
                supercat = coco.__dict__["cats"][cat_id]["supercategory"]
                cat = coco.__dict__["cats"][cat_id]["name"]
                data["supercats"].append(supercat)
                data["cats"].append(cat)
                continue
        data["supercats"] = np.unique(data["supercats"])
        data["cats"] = np.unique(data["cats"])
        # print(data)
        all_data.append(data)
    return all_data


# all_data = look_up_cats(flat_img_ids, coco)


def get_anns(img_ids, coco):
    all_data = []
    for img_id in tqdm(img_ids, desc="looking up cats"):
        data = {}
        data["id"] = img_id
        data["supercats"] = []
        data["cats"] = []
        data["areas"] = []
        # print("looking for image", img_id)
        for anns in coco.__dict__["imgToAnns"][img_id]:
            cat_id = anns["category_id"]
            supercat = coco.__dict__["cats"][cat_id]["supercategory"]
            cat = coco.__dict__["cats"][cat_id]["name"]
            data["supercats"].append(supercat)
            data["cats"].append(cat)
            data["areas"].append(anns["area"])
        try:
            i = np.argmax(data["areas"])
            data["supercat"] = data["supercats"][i]
            data["cat"] = data["cats"][i]
        except:
            data["supercat"] = "unknown"
            data["cat"] = "unknown"
        all_data.append(data)
    return all_data


all_data = get_anns(flat_img_ids, coco)
all_data = pd.DataFrame(all_data)
all_data.to_pickle(os.path.join(NSD_DIR, "info", " labels.pkl"))

### create supercat matrix of shape (8 subjects, 30000 trials)
supercat = np.zeros(img_ids.shape, dtype="object")
for i in tqdm(range(img_ids.shape[0]), desc="i", position=0):
    for j in tqdm(range(img_ids.shape[1]), desc="j", position=1):
        img_id = img_ids[i, j]
        supercat[i, j] = all_data[all_data["id"] == img_id]["supercat"].values[
            0
        ]
np.save(os.path.join(NSD_DIR, "curated_3mm", "supercat_labels.npy"), supercat)

### create run number matrices
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
np.save(os.path.join(NSD_DIR, "curated_3mm", "runs.npy"), run_numbers)


### start the process to remove unknowns and trials not done by all subjects
# keep only first 22500 images per subject, because the rest are not done by all subjects
# so set the rest to unknown to remove at a later stage
supercat[:, 22500:] = "unknown"
# check class representation and get number of unknowns per subject
unknown_counts = []
for i in tqdm(range(img_ids.shape[0]), desc="i", position=0):
    unique, counts = np.unique(supercat[i, :], return_counts=True)
    print("sub", i + 1)
    cat_counts = dict(zip(unique, counts))
    print(cat_counts)
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
np.save(os.path.join(NSD_DIR, "curated_3mm", "unknown_mask.npy"), unknown_mask)

# load mask
unknown_mask = np.load(
    os.path.join(NSD_DIR, "curated_3mm", "unknown_mask.npy")
)
# load supercat labels
supercat = np.load(os.path.join(NSD_DIR, "curated_3mm", "supercat_labels.npy"))
# load run numbers
run_numbers = np.load(os.path.join(NSD_DIR, "curated_3mm", "runs.npy"))
### concatenate and remove nifti volumes corresponding to unknown supercats
three_mm = os.path.join(NSD_DIR, "3mm")
subs = os.listdir(three_mm)
out_subs = [f"sub-{i:02d}" for i in range(len(subs))]
for i, sub in tqdm(enumerate(subs), desc="sub", position=0):
    sub_dir = os.path.join(three_mm, sub)
    niftis = glob(os.path.join(sub_dir, "betas*.nii.gz"))
    print("concatenating...")
    concatenated = image.concat_imgs(niftis)
    print(concatenated.shape)
    print("masking...")
    concatenated = image.index_img(concatenated, np.invert(unknown_mask[i]))
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


### get all nifti file paths
one_mm = os.path.join(NSD_DIR, "1mm")
three_mm = os.path.join(NSD_DIR, "3mm")
subs = os.listdir(one_mm)
niftis = []
sub_all = []
for sub in subs:
    sub_dir = os.path.join(one_mm, sub)
    os.makedirs(os.path.join(three_mm, sub), exist_ok=True)
    niftis_ = glob(os.path.join(sub_dir, "betas*.nii.gz"))
    for nifti in niftis_:
        niftis.append(os.path.join(sub_dir, nifti))
        sub_all.append(sub)
gm_mask = load_mni152_gm_mask()


### downsample to 3mm resolution
# first resample and then mask
def resample_mask(three_mm, nifti, sub, gm_mask):
    print(f"resampling {nifti} to 3mm")
    resampled_img = image.resample_img(nifti, target_affine=np.diag((3, 3, 3)))
    print(f"resampling mask to 3mm {nifti}")
    gm_mask = image.resample_to_img(
        gm_mask, resampled_img, interpolation="nearest"
    )
    print(f"masking {nifti}")
    masker = NiftiMasker(mask_img=gm_mask)
    resampled_img = masker.fit_transform(resampled_img)
    print(f"unmasking {nifti}")
    resampled_img = unmask(resampled_img, gm_mask)
    nifti_file = os.path.basename(nifti)
    print(f"saving {os.path.join(three_mm, sub, nifti_file)}")
    resampled_img.to_filename(os.path.join(three_mm, sub, nifti_file))


### downsample to 3mm resolution
# first mask and then resample
def mask_resample(three_mm, nifti, sub, gm_mask):
    print(f"resampling mask to 1mm {nifti}")
    gm_mask = image.resample_to_img(gm_mask, nifti, interpolation="nearest")
    print(f"masking {nifti}")
    masker = NiftiMasker(mask_img=gm_mask)
    masked_img = masker.fit_transform(nifti)
    print(f"unmasking {nifti}")
    masked_img = unmask(masked_img, gm_mask)
    print(f"resampling {nifti} to 3mm")
    masked_img = image.resample_img(
        masked_img, target_affine=np.diag((3, 3, 3))
    )
    nifti_file = os.path.basename(nifti)
    print(f"saving {os.path.join(three_mm, sub, nifti_file)}")
    masked_img.to_filename(os.path.join(three_mm, sub, nifti_file))


### get all nifti file paths
one_mm = os.path.join(NSD_DIR, "1mm")
three_mm = os.path.join(NSD_DIR, "tmp")
subs = os.listdir(one_mm)
niftis = []
sub_all = []
for sub in subs:
    sub_dir = os.path.join(one_mm, sub)
    os.makedirs(os.path.join(three_mm, sub), exist_ok=True)
    niftis_ = glob(os.path.join(sub_dir, "betas*.nii.gz"))
    for nifti in niftis_:
        niftis.append(os.path.join(sub_dir, nifti))
        sub_all.append(sub)
gm_mask = load_mni152_gm_mask()

x = Parallel(n_jobs=10, verbose=11, backend="loky")(
    delayed(resample_mask)(three_mm, nifti, sub, gm_mask)
    for nifti, sub in zip(niftis, sub_all)
)


mask_first = (
    "/storage/store3/data/natural_scenes/tmp/subj01/betas_session01.nii.gz"
)
resample_first = (
    "/storage/store3/data/natural_scenes/3mm/subj01/betas_session01.nii.gz"
)
mask_first = image.index_img(mask_first, 0).get_fdata().flatten()
resample_first = image.index_img(resample_first, 0).get_fdata().flatten()
mask_first_thresh = mask_first == 0
resample_first_thresh = resample_first == 0


plt.hist(
    mask_first[mask_first_thresh],
    bins=50,
    density=True,
    range=(-500, 500),
)
plt.hist(
    resample_first[resample_first_thresh],
    bins=50,
    density=True,
    range=(-500, 500),
)
plt.savefig("hist.png", bbox_inches="tight")
plt.close("all")
