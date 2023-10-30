"""This script gets the coco category labels for each image in NSD."""

import scipy.io
import os
import pandas as pd
from pycocotools.coco import COCO
import numpy as np
from tqdm import tqdm


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
        # we want the category with the largest area
        try:
            i = np.argmax(data["areas"])
            data["supercat"] = data["supercats"][i]
            data["cat"] = data["cats"][i]
        except:
            data["supercat"] = "unknown"
            data["cat"] = "unknown"
        all_data.append(data)
    return all_data


NSD_DIR = "/storage/store3/data/natural_scenes/"

exp_design = scipy.io.loadmat(
    os.path.join(NSD_DIR, "info", "nsd_expdesign.mat")
)

img_ids = exp_design["subjectim"][
    :, exp_design["masterordering"] - 1
].squeeze()

merged_file = os.path.join(
    NSD_DIR, "info", "coco_annotations", "merged_instances_train_val_2017.json"
)
coco = COCO(merged_file)

flat_img_ids = np.unique(img_ids.flatten())

all_data = get_anns(flat_img_ids, coco)
all_data = pd.DataFrame(all_data)
all_data.to_pickle(
    os.path.join(NSD_DIR, "info", "coco_annotations", "coco_categories.pkl")
)
