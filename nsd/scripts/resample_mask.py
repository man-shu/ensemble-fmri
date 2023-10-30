"""This script resamples the 1mm files to 3mm, 
and then applies MNI gm mask to remove all extra cortical voxels"""

import os
import numpy as np
from joblib import Parallel, delayed
from nilearn import image
from glob import glob
from nilearn.datasets import load_mni152_gm_mask
from nilearn.maskers import NiftiMasker
from nilearn.masking import unmask


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


NSD_DIR = "/storage/store3/data/natural_scenes/"

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

x = Parallel(n_jobs=10, verbose=11, backend="loky")(
    delayed(resample_mask)(three_mm, nifti, sub, gm_mask)
    for nifti, sub in zip(niftis, sub_all)
)
