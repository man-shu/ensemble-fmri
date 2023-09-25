"""
This script aims a per-event study for the RSVP language protocol of IBC.

to get ibc_public: install git@github.com:hbp-brain-charting/public_analysis_code.git

Author: Bertrand Thirion, Thomas Bazeille
"""
import glob
import os
import numpy as np
from nilearn.input_data import NiftiMasker
import ibc_public.utils_data
from nilearn.glm.first_level import (
    make_first_level_design_matrix,FirstLevelModel)
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut

MAIN_CONDITIONS = ['complex', 'simple', 'consonant_strings', 'word_list',
                   'jabberwocky', 'pseudoword_list']
# MAIN_CONDITIONS = ['complex', 'simple']


def make_dmtx(events, fmri, confounds=None, t_r=2):
    from pandas import read_csv
    import nibabel as nib
    n_scans = nib.load(fmri).shape[3]
    frame_times = np.arange(n_scans) * t_r
    paradigm = read_csv(events, sep='\t')
    complexs = ['complex_sentence_objclef', 'complex_sentence_objrel',
                'complex_sentence_subjrel']
    simples = ['simple_sentence_adj', 'simple_sentence_coord',
               'simple_sentence_cvp']
    for complex_ in complexs:
        paradigm = paradigm.replace(complex_, 'complex')
    for simple_ in simples:
        paradigm = paradigm.replace(simple_, 'simple')
    trial_type = paradigm.trial_type.values
    for condition in MAIN_CONDITIONS:
        trial_type[trial_type == condition] = ['%s_%02d' % (condition, i)
                                               for i in range(10)]
    paradigm.trial_type = trial_type
    motion = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']
    dmtx = make_first_level_design_matrix(
        frame_times, events=paradigm, hrf_model='spm',
        add_regs=np.loadtxt(confounds), add_reg_names=motion)
    # print(dmtx.columns)
    return dmtx


if 1:  # work at Neurospin
    ibc = '/neurospin/ibc/derivatives'
else:  # work on drago
    ibc = '/storage/store/data/ibc/derivatives/'  # Check

# obtain a grey matter mask
_package_directory = os.path.dirname(
    os.path.abspath(ibc_public.utils_data.__file__))
mask_gm = os.path.join(
    _package_directory, '../ibc_data', 'gm_mask_1_5mm.nii.gz')

# sessions + subject data
session_file = os.path.join(
    _package_directory, '../ibc_data', 'sessions.csv')
subject_session = ibc_public.utils_data.get_subject_session('rsvp-language')
write_dir = '/neurospin/tmp/bthirion/rsvp_trial'
if not os.path.exists(write_dir):
    os.mkdir(write_dir)

# set up a masker
model = FirstLevelModel(mask_img=mask_gm, t_r=2., high_pass=1./128,
                        smoothing_fwhm=5)
masker = NiftiMasker(mask_img=mask_gm, t_r=2., high_pass=1./128,
                     standardize=True, detrend=True, smoothing_fwhm=5)

scores = []
for subject, session in subject_session:
    # fetch the data
    data_path = os.path.join(ibc, subject, session, 'func')
    fmri = sorted(glob.glob(os.path.join(data_path, 'w*RSVPLanguage*')))
    confounds = sorted(glob.glob(os.path.join(data_path, 'rp_*RSVPLanguage*')))
    events = sorted(glob.glob(os.path.join(data_path,
                                           '*RSVPLanguage*_events.tsv')))
    Y = []
    X = []
    labels = []
    n_sessions = len(fmri)
    local_dir = os.path.join(write_dir, subject)
    if not os.path.exists(local_dir):
        os.mkdir(local_dir)

    for i in range(n_sessions):
        dmtx = make_dmtx(events[i], fmri[i], confounds[i])
        model.fit(fmri[i], design_matrices=[dmtx])
        # contrasts = np.eye(dmtx.shape[1])
        for trial in dmtx.columns:
            name = trial[:-3]
            if name in MAIN_CONDITIONS:
                z = model.compute_contrast(dmtx.columns == trial)
                z.to_filename(os.path.join(local_dir, 'session_%i_%s.nii.gz') %
                              (i, trial))
                z_ = z.get_data()
                Y.append(z_[z_ != 0])
                X.append(name)
                labels.append(i)

    X = np.array(X)
    Y = np.array(Y)
    labels = np.array(labels)
    clf = LinearSVC()
    cv = LeaveOneGroupOut()
    score = cross_val_score(clf, Y, X, groups=labels, cv=cv, n_jobs=3)
    print(score.mean())
    scores.append(score.mean())
