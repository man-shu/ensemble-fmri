"""
This script aims a per-event study for the RSVP language protocol of IBC.

to get ibc_public: install git@github.com:hbp-brain-charting/public_analysis_code.git

Author: Bertrand Thirion, Thomas Bazeille
"""
import glob
import os
import numpy as np
from nilearn.input_data import NiftiMasker
from nilearn import image
import ibc_public.utils_data
from nilearn.glm.first_level import (
    make_first_level_design_matrix,
    FirstLevelModel,
)
from tqdm import tqdm

MAIN_CONDITIONS = [
    "anger",
    "pride",
    "joy",
    "contempt",
    "neutral",
]
# MAIN_CONDITIONS = ['complex', 'simple']


def make_dmtx(events, fmri, confounds=None, t_r=2):
    from pandas import read_csv
    import nibabel as nib

    n_scans = nib.load(fmri).shape[3]
    frame_times = np.arange(n_scans) * t_r
    paradigm = read_csv(events, sep="\t")
    trial_type = paradigm.trial_type.values
    for condition in MAIN_CONDITIONS:
        trial_type[trial_type == condition] = [
            "%s_%02d" % (condition, i) for i in range(10)
        ]
    paradigm.trial_type = trial_type
    motion = ["tx", "ty", "tz", "rx", "ry", "rz"]
    dmtx = make_first_level_design_matrix(
        frame_times,
        events=paradigm,
        hrf_model="spm",
        add_regs=np.loadtxt(confounds),
        add_reg_names=motion,
    )
    # print(dmtx.columns)
    return dmtx


ibc = "/ibc_data/preprocessed/"

# obtain a grey matter mask
_package_directory = os.path.dirname(
    os.path.abspath(ibc_public.utils_data.__file__)
)
mask_gm = os.path.join(
    _package_directory, "../ibc_data", "gm_mask_1_5mm.nii.gz"
)

# sessions + subject data
session_file = os.path.join(_package_directory, "../ibc_data", "sessions.csv")
subject_session = ibc_public.utils_data.get_subject_session("rsvp-language")
write_dir = "rsvp_trial"
if not os.path.exists(write_dir):
    os.mkdir(write_dir)

# set up a masker
model = FirstLevelModel(
    mask_img=mask_gm, t_r=2.0, high_pass=1.0 / 128, smoothing_fwhm=5
)
masker = NiftiMasker(
    mask_img=mask_gm,
    t_r=2.0,
    high_pass=1.0 / 128,
    standardize=True,
    detrend=True,
    smoothing_fwhm=5,
)

scores = []
for subject, session in tqdm(subject_session):
    # fetch the data
    data_path = os.path.join(ibc, subject, session, "func")
    fmri = sorted(
        glob.glob(os.path.join(data_path, "*RSVPLanguage*_bold.nii.gz"))
    )
    confounds = sorted(
        glob.glob(
            os.path.join(data_path, "*RSVPLanguage*_confounds_timeseries.tsv")
        )
    )
    events = sorted(
        glob.glob(os.path.join(data_path, "*RSVPLanguage*_events.tsv"))
    )
    Y = []
    X = []
    labels = []
    n_sessions = len(fmri)
    local_dir = os.path.join(write_dir)
    if not os.path.exists(local_dir):
        os.mkdir(local_dir)

    all_sessions = []
    for i in range(n_sessions):
        dmtx = make_dmtx(events[i], fmri[i], confounds[i])
        model.fit(fmri[i], design_matrices=[dmtx])
        # contrasts = np.eye(dmtx.shape[1])
        for trial in dmtx.columns:
            name = trial[:-3]
            if name in MAIN_CONDITIONS:
                z = model.compute_contrast(dmtx.columns == trial)
                all_sessions.append(z)
    z = image.concat_imgs(all_sessions)
    print(z.shape)
    one_point_five_mm_dir = os.makedirs(
        os.path.join(local_dir, "1_5mm"), exist_ok=True
    )
    z.to_filename(os.path.join(one_point_five_mm_dir, f"{subject}.nii.gz"))
    # downsample to 3mm
    three_mm_dir = os.makedirs(os.path.join(local_dir, "3mm"), exist_ok=True)
    z = image.resample_img(z, target_affine=np.diag((3, 3, 3)))
    z.to_filename(os.path.join(three_mm_dir, f"{subject}.nii.gz"))
