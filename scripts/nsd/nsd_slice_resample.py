"""This script retains first n trials and resamples them from 1mm to 3mm"""

import os
import numpy as np

# from joblib import Parallel, delayed
from nilearn import image
from glob import glob
from tqdm import tqdm

N_TRIALS = 22500


### slice and downsample to 3mm resolution
def slice_resample(three_mm, nifti, sub):
    image.resample_img(
        image.index_img(nifti, slice(0, N_TRIALS)),
        target_affine=np.diag((3, 3, 3)),
    ).to_filename(os.path.join(three_mm, f"{sub}.nii.gz"))


ROOT = "/storage/store2/work/haggarwa/retreat_2023/data/"
IN_DIR = os.path.join(ROOT, "natural_scenes")
OUT_DIR = os.path.join(ROOT, "nsd")
### get all nifti file paths
one_mm = os.path.join(IN_DIR, "1mm")
three_mm = os.path.join(OUT_DIR, f"3mm_{N_TRIALS}trials")
os.makedirs(three_mm, exist_ok=True)
subs = os.listdir(one_mm)

for sub in tqdm(subs):
    niftis = []
    sub_dir = os.path.join(one_mm, sub)
    niftis_ = glob(os.path.join(sub_dir, "betas*.nii.gz"))
    for nifti in niftis_:
        niftis.append(os.path.join(sub_dir, nifti))
    slice_resample(three_mm, image.concat_imgs(niftis, verbose=11), sub)
