"""This script fetches the names of the files presented in BOLD5000, filters COCO images, gets their index of presentation and uses that as a mask to filter the GLM betas."""

import os
import pandas as pd
import numpy as np
from glob import glob
from nilearn import image
from tqdm import tqdm
from pycocotools.coco import COCO


def get_labels(img_ids, coco):
    labels = []
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
        # we want the category with the largest area
        try:
            i = np.argmax(data["areas"])
            data["supercat"] = data["supercats"][i]
            data["cat"] = data["cats"][i]
        except:
            data["supercat"] = "unknown"
            data["cat"] = "unknown"
        labels.append(data)
    return labels


BOLD_DIR = "/storage/store3/data/bold5000/"

OUTPUT_DIR = "/storage/store2/work/haggarwa/retreat_2023/data/bold/"
three_mm_dir = os.path.join(OUTPUT_DIR, "3mm")
os.makedirs(three_mm_dir, exist_ok=True)
two_mm_dir = os.path.join(OUTPUT_DIR, "2mm")
os.makedirs(two_mm_dir, exist_ok=True)

coco = COCO(
    os.path.join(
        BOLD_DIR,
        "info",
        "coco_annotations",
        "instances_train2014.json",
    )
)

subs = ["CSI1", "CSI2", "CSI3", "CSI4"]

# create a reference list of trials that are present in all subjects
all_trials = []
for sub in subs:
    trials = pd.read_csv(
        os.path.join(BOLD_DIR, "denoised_3mm", f"{sub}_imgnames.txt"),
        header=None,
    )
    trials = list(trials[trials[0].str.contains("COCO_")].drop_duplicates()[0])
    all_trials.append(trials)

ref_trials = list(
    set(all_trials[0])
    & set(all_trials[1])
    & set(all_trials[2])
    & set(all_trials[3])
)

# remove the ones that are not in all subjects
# also get the category labels
for sub in subs:
    trials = pd.read_csv(
        os.path.join(BOLD_DIR, "denoised_3mm", f"{sub}_imgnames.txt"),
        header=None,
    )
    trials = trials[trials[0].str.contains("COCO_")].drop_duplicates()[0]
    trials = trials[trials.isin(ref_trials)]

    img_ids = list(
        trials.str.replace(".jpg", "").str.split("_").str[-1].astype(int)
    )

    labels = get_labels(img_ids, coco)
    labels = pd.DataFrame(labels)

    # print(labels.value_counts("supercat"))

    # now only keep the four most represented categories:
    # person, vehicle, furniture, animal
    # furniture is the least represented with only 149 images
    # so we only keep 149 images from the other categories to balance the dataset
    # also get the indices of the images that we want to keep
    cats_to_keep = ["person", "vehicle", "furniture", "animal"]
    n_samples_per_cat = 149
    indices = []
    for cat in cats_to_keep:
        samples_to_keep = labels[labels["supercat"] == cat].sample(
            n_samples_per_cat, random_state=42
        )
        idx = samples_to_keep.index.values
        # print(idx, len(idx))
        indices.extend(idx)

    indices = np.array(indices)
    indices = np.sort(indices)

    labels = labels.iloc[indices]["supercat"]
    labels.to_csv(
        os.path.join(three_mm_dir, f"{sub}_labels.csv"),
        index=False,
        header=None,
    )

    # now we have the indices of the images that we want to keep
    niftis = glob(os.path.join(BOLD_DIR, "denoised_3mm", f"{sub}*GLM*.nii.gz"))
    concat = image.concat_imgs(niftis)
    concat = image.index_img(concat, indices)
    print(concat.shape)
    concat.to_filename(os.path.join(three_mm_dir, f"{sub}.nii.gz"))
