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
    aomic = os.path.join(ROOT, "data", "aomic")
    derivatives = os.path.join(aomic, "ds002785", "derivatives", "fmriprep")
    source_dir = os.path.join(aomic, "ds002785")

    # hand conditions
    main_conditions = [
        "congruent_male_male",
        "congruent_female_female",
        "incongruent_male_female",
        "incongruent_female_male",
    ]

    # TR
    t_r = 2.0

    # voxels are already 3mm, 3mm x 3mm x 3.3mm to be exact
    # so we don't need to resample
    three_mm_dir = os.path.join(ROOT, "data", "aomic_gstroop", "3mm")
    os.makedirs(three_mm_dir, exist_ok=True)

    # design matrix plots dir
    design_matrix_dir = os.path.join(
        ROOT, "data", "aomic_gstroop", "design_matrices"
    )
    os.makedirs(design_matrix_dir, exist_ok=True)

    ## get all fmri, events, mask paths
    fmri = sorted(
        glob(
            os.path.join(
                derivatives,
                "sub-*",
                "func",
                "*task-gstroop*MNI*bold.nii.gz",
            )
        )
    )
    subjects = [f.split(os.sep)[11] for f in fmri]
    events = [
        glob(
            os.path.join(source_dir, sub, "func", "*task-gstroop*events.tsv")
        )[0]
        for sub in subjects
    ]
    masks = [
        glob(
            os.path.join(
                derivatives,
                sub,
                "func",
                "*task-gstroop*MNI*brain_mask.nii.gz",
            )
        )[0]
        for sub in subjects
    ]

    ## keep only first n subjects
    # cutoff = 1
    # subjects = subjects[:cutoff]
    # fmri = fmri[:cutoff]
    # events = events[:cutoff]
    # masks = masks[:cutoff]

    all_files = pd.DataFrame(
        {
            "subject": subjects,
            "fmri": fmri,
            "events": events,
            "mask": masks,
        }
    )

    # iterate over all files, subject by subject
    for i, files in tqdm(all_files.iterrows(), total=all_files.shape[0]):
        subject = files["subject"]
        events_df = pd.read_csv(files["events"], sep="\t")
        fmri_file = files["fmri"]
        mask_file = files["mask"]

        # combine trial_type, img_gender and word_gender
        events_df["trial_type"] = (
            events_df["trial_type"]
            + "_"
            + events_df["img_gender"]
            + "_"
            + events_df["word_gender"]
        )

        # drop nans or non-hand trials, if any
        events_df = events_df.loc[
            events_df["trial_type"].isin(main_conditions)
        ]

        # only keep onset, duration, trial_type
        events_df = events_df[["onset", "duration", "trial_type"]]
        events_df.reset_index(drop=True, inplace=True)

        # skip subject if only one condition
        if len(events_df["trial_type"].unique()) < 2:
            continue
        # skip subject if not enough trials
        if len(events_df) < 10:
            continue

        # Load the confounds
        confounds, _ = load_confounds(fmri_file)

        # calculate the number of scans, and the frame times
        n_scans = nib.load(fmri_file).shape[3]
        frame_times = np.arange(n_scans) * t_r

        # Loop through the trials of interest and transform the DataFrame for LSS
        labels_maps = Parallel(n_jobs=50, verbose=11)(
            delayed(get_trial_wise_maps)(
                fmri_file,
                mask_file,
                events_df,
                confounds,
                t_r,
                i_trial,
                subject,
                design_matrix_dir,
            )
            for i_trial in range(events_df.shape[0])
        )

        # Unpack the labels and maps
        labels, maps = zip(*labels_maps)

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
