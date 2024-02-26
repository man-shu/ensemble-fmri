from glob import glob
import os
import numpy as np
from nilearn import image
from nilearn.interfaces.fmriprep import load_confounds
from nilearn.glm.first_level import FirstLevelModel
from nilearn.plotting import plot_design_matrix
from tqdm import tqdm
import pandas as pd
from joblib import Parallel, delayed
import nibabel as nib
import ibc_public.utils_data
import warnings

warnings.filterwarnings("ignore")


def lss_transformer(df, row_number):
    """Label one trial for one LSS model.

    Parameters
    ----------
    df : pandas.DataFrame
        BIDS-compliant events file information.
    row_number : int
        Row number in the DataFrame.
        This indexes the trial that will be isolated.

    Returns
    -------
    df : pandas.DataFrame
        Update events information, with the select trial's trial type isolated.
    trial_name : str
        Name of the isolated trial's trial type.
    """
    df = df.copy()

    # Determine which number trial it is *within the condition*
    trial_condition = df.loc[row_number, "trial_type"]
    trial_type_series = df["trial_type"]
    trial_type_series = trial_type_series.loc[
        trial_type_series == trial_condition
    ]
    trial_type_list = trial_type_series.index.tolist()
    trial_number = trial_type_list.index(row_number)

    # We use a unique delimiter here (``__``) that shouldn't be in the
    # original condition names.
    # Technically, all you need is for the requested trial to have a unique
    # 'trial_type' *within* the dataframe, rather than across models.
    # However, we may want to have meaningful 'trial_type's (e.g., 'Left_001')
    # across models, so that you could track individual trials across models.
    trial_name = f"{trial_condition}__{trial_number:03d}"
    df.loc[row_number, "trial_type"] = trial_name
    return df, trial_name


def get_trial_wise_maps(
    fmri_file,
    mask_file,
    events_df,
    confounds,
    t_r,
    i_trial,
    subject,
    design_matrix_dir,
):
    lss_events_df, trial_condition = lss_transformer(events_df, i_trial)

    # Compute and collect beta maps
    lss_glm = FirstLevelModel(
        mask_img=mask_file,
        t_r=t_r,
        high_pass=1.0 / 128,
        smoothing_fwhm=5,
    )
    lss_glm.fit(fmri_file, lss_events_df, confounds=confounds)

    # Save the design matrix plot
    design_matrix_plot = os.path.join(
        design_matrix_dir,
        f"design_matrix_{subject}_{trial_condition}.png",
    )
    plot_design_matrix(
        lss_glm.design_matrices_[0],
        output_file=design_matrix_plot,
    )

    beta_map = lss_glm.compute_contrast(
        trial_condition,
        output_type="effect_size",
    )

    # Drop the trial number from the condition name to get the original name
    condition_name = trial_condition.split("__")[0]

    return (condition_name, beta_map)


if __name__ == "__main__":
    ROOT = "/storage/store2/work/haggarwa/retreat_2023"
    ibc = os.path.join(ROOT, "data", "ibc")
    derivatives = os.path.join(ibc, "derivatives")

    # obtain a grey matter mask
    _package_directory = os.path.dirname(
        os.path.abspath(ibc_public.utils_data.__file__)
    )
    mask_gm = os.path.join(
        _package_directory, "../ibc_data", "gm_mask_3mm.nii.gz"
    )

    # sessions + subject data
    session_file = os.path.join(
        _package_directory, "../ibc_data", "sessions.csv"
    )
    subject_session = ibc_public.utils_data.get_subject_session(
        "rsvp-language"
    )

    three_mm_dir = os.path.join(ROOT, "data", "rsvp", "3mm")
    os.makedirs(three_mm_dir, exist_ok=True)

    # hand conditions
    main_conditions = [
        "complex",
        "simple",
        "consonant_strings",
        "word_list",
        "jabberwocky",
        "pseudoword_list",
    ]

    # TR
    t_r = 2.0

    # design matrix plots dir
    design_matrix_dir = os.path.join(ROOT, "data", "rsvp", "design_matrices")
    os.makedirs(design_matrix_dir, exist_ok=True)

    for subject, session in tqdm(subject_session):
        # fetch the data
        data_path = os.path.join(derivatives, subject, session, "func")
        fmri = sorted(
            glob(os.path.join(data_path, "w*RSVPLanguage*_bold.nii.gz"))
        )
        confounds = sorted(
            glob(os.path.join(data_path, "rp_*RSVPLanguage*.txt"))
        )
        events = sorted(
            glob(os.path.join(data_path, "*RSVPLanguage*_events.tsv"))
        )
        Y = []
        X = []
        labels = []
        n_sessions = len(fmri)

        all_sessions = []
        for i in range(n_sessions):
            fmri_file = fmri[i]
            fmri_file = image.resample_img(
                fmri_file, target_affine=np.diag((3, 3, 3))
            )

            confound_df = np.loadtxt(confounds[i])
            confound_df = pd.DataFrame(confound_df)

            events_df = pd.read_csv(events[i], sep="\t")
            complexs = [
                "complex_sentence_objclef",
                "complex_sentence_objrel",
                "complex_sentence_subjrel",
            ]
            simples = [
                "simple_sentence_adj",
                "simple_sentence_coord",
                "simple_sentence_cvp",
            ]
            for complex_ in complexs:
                events_df = events_df.replace(complex_, "complex")
            for simple_ in simples:
                events_df = events_df.replace(simple_, "simple")

            # only keep main conditions
            events_df = events_df.loc[
                events_df["trial_type"].isin(main_conditions)
            ]
            events_df.reset_index(inplace=True, drop=True)

            # Loop through the trials of interest and transform the DataFrame for LSS
            labels_maps = Parallel(n_jobs=50, verbose=11)(
                delayed(get_trial_wise_maps)(
                    fmri_file,
                    mask_gm,
                    events_df,
                    confound_df,
                    t_r,
                    i_trial,
                    subject,
                    design_matrix_dir,
                )
                for i_trial in range(events_df.shape[0])
            )

            all_sessions.extend(labels_maps)

        # Unpack the labels and maps
        labels, maps = zip(*all_sessions)

        # Save the beta maps
        subject_map_file = os.path.join(
            three_mm_dir,
            f"{subject}.nii.gz",
        )
        subject_map = image.concat_imgs(maps)
        subject_map.to_filename(subject_map_file)

        # labels for the beta maps
        label_file = os.path.join(three_mm_dir, f"{subject}_labels.csv")
        pd.Series(labels).to_csv(label_file, index=None, header=None)
