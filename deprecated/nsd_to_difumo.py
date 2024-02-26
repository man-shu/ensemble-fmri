from nilearn import datasets, maskers
import numpy as np
import os
from tqdm import tqdm
from glob import glob

DATA_ROOT = "/storage/store2/work/haggarwa/retreat_2023/data/"
OUT_ROOT = "/storage/store2/work/haggarwa/retreat_2023"

dataset = "nsd"

# input data root path
data_dir = os.path.join(DATA_ROOT, dataset)
data_resolution = "3mm"  # or 1_5mm
nifti_dir = os.path.join(data_dir, data_resolution)
classifiers = ["LinearSVC", "RandomForest"]
# output results path
results_dir = "difumo"
results_dir = os.path.join(data_dir, results_dir)
os.makedirs(results_dir, exist_ok=True)
# create empty dictionary to store data
data = dict(responses=[], conditions=[], runs=[], subjects=[])
imgs = glob(os.path.join(nifti_dir, "*.nii.gz"))
subjects = [os.path.basename(img).split(".")[0] for img in imgs]
# parcellate data with atlas
# get difumo atlas
atlas = datasets.fetch_atlas_difumo(
    dimension=1024, resolution_mm=3, data_dir=DATA_ROOT
)
# parcellate data
print("Parcellating...")
# make appropriate subject, run, condition labels
for i, subject in tqdm(enumerate(subjects), total=len(subjects)):
    masker = maskers.NiftiMapsMasker(
        maps_img=atlas.maps, memory=DATA_ROOT, verbose=11, n_jobs=20
    )
    np.save(
        os.path.join(results_dir, f"{subject}.npy"),
        masker.fit_transform(imgs[i]),
    )
