"""This script retains first n trials and resamples them from 1mm to 3mm"""

import os
import numpy as np

from joblib import Parallel, delayed
from nilearn import image
from glob import glob
from tqdm import tqdm


### downsample to 3mm resolution
def resample(sub):
    sub_dir = os.path.join(one_mm, sub)
    out_sub_dir = os.path.join(three_mm, sub)
    os.makedirs(os.path.join(three_mm, sub), exist_ok=True)
    niftis = glob(os.path.join(sub_dir, "betas*.nii.gz"))
    for nifti in tqdm(niftis):
        session = nifti.split("/")[-1]
        if os.path.exists(os.path.join(out_sub_dir, session)):
            continue
        else:
            image.resample_img(
                nifti, target_affine=np.diag((3, 3, 3))
            ).to_filename(os.path.join(out_sub_dir, session))


ROOT = "/storage/store2/work/haggarwa/retreat_2023/data/"
IN_DIR = os.path.join(ROOT, "natural_scenes")
OUT_DIR = os.path.join(ROOT, "natural_scenes")
### get all nifti file paths
one_mm = os.path.join(IN_DIR, "1mm")
three_mm = os.path.join(OUT_DIR, "3mm_corrected")
os.makedirs(three_mm, exist_ok=True)
subs = os.listdir(one_mm)

Parallel(n_jobs=8)(delayed(resample)(sub) for sub in subs)
