import os
import numpy as np
from nilearn import image
from nilearn.glm.first_level import FirstLevelModel
from nilearn.plotting import plot_design_matrix
from tqdm import tqdm
import pandas as pd
from joblib import Parallel, delayed
import nibabel as nib
from nilearn.datasets import load_mni152_gm_mask
import time


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
    run,
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
    print("fitting lss glm...")
    t0 = time.time()
    lss_glm.fit(fmri_file, lss_events_df, confounds=confounds)
    t1 = time.time()
    print(f"done fitting lss glm in {t1-t0}...")

    # Save the design matrix plot
    design_matrix_plot = os.path.join(
        design_matrix_dir,
        f"design_matrix_{subject}_{trial_condition}.png",
    )
    plot_design_matrix(
        lss_glm.design_matrices_[0],
        output_file=design_matrix_plot,
    )

    print("computing contrast...")
    beta_map = lss_glm.compute_contrast(
        trial_condition,
        output_type="effect_size",
    )

    # Drop the trial number from the condition name to get the original name
    condition_name = trial_condition.split("__")[0]

    return (condition_name, run, beta_map)


if __name__ == "__main__":
    ROOT = "/storage/store2/work/haggarwa/retreat_2023"
    hcp = os.path.join(ROOT, "data", "HCP900")

    task = "GAMBLING"

    # get all subjects
    dirs = os.listdir(hcp)
    all_subjects = []
    for sub in dirs:
        try:
            int(sub)
            all_subjects.append(sub)
        except ValueError:
            print(sub, "is not a subject directory")
            continue
    # total 1137 subjects
    print(len(all_subjects))

    # keep subjects where GAMBLING task exists
    subjects = []
    for sub in all_subjects:
        if not os.path.exists(
            os.path.join(
                hcp,
                sub,
                "MNINonLinear",
                "Results",
                f"tfMRI_{task}_RL",
            )
        ):
            print(f"no {task} for {sub}")
            continue
        else:
            subjects.append(sub)
    # GAMBLING task exists for 568 subjects
    print(len(subjects))

    # get all fmri, event files paths
    all_files = []
    for sub in subjects:
        for run in ["RL", "LR"]:
            fmri = os.path.join(
                hcp,
                sub,
                "MNINonLinear",
                "Results",
                f"tfMRI_{task}_{run}",
                f"tfMRI_{task}_{run}.nii.gz",
            )
            event_loss = os.path.join(
                hcp,
                sub,
                "MNINonLinear",
                "Results",
                f"tfMRI_{task}_{run}",
                "EVs",
                "loss_event.txt",
            )
            event_win = os.path.join(
                hcp,
                sub,
                "MNINonLinear",
                "Results",
                f"tfMRI_{task}_{run}",
                "EVs",
                "win_event.txt",
            )
            movement = os.path.join(
                hcp,
                sub,
                "MNINonLinear",
                "Results",
                f"tfMRI_{task}_{run}",
                "Movement_Regressors.txt",
            )
            derivative = os.path.join(
                hcp,
                sub,
                "MNINonLinear",
                "Results",
                f"tfMRI_{task}_{run}",
                "Movement_Regressors_dt.txt",
            )
            files = {
                "fmri": fmri,
                "loss": event_loss,
                "win": event_win,
                "subject": sub,
                "run": run,
                "movement": movement,
                "derivative": derivative,
            }
            all_files.append(files)
    all_files = pd.DataFrame(all_files)

    # loss win conditions
    main_conditions = [
        "loss",
        "win",
    ]
    # TR
    t_r = 0.72

    # mask
    mask_file = load_mni152_gm_mask()

    three_mm_dir = os.path.join(ROOT, "data", "hcp_gambling", "3mm")
    os.makedirs(three_mm_dir, exist_ok=True)

    # design matrix plots dir
    design_matrix_dir = os.path.join(
        ROOT, "data", "hcp_gambling", "design_matrices"
    )
    os.makedirs(design_matrix_dir, exist_ok=True)

    ## keep only first n subjects
    start_sub = 101
    end_sub = 140
    subjects_to_keep = all_files["subject"].unique()[start_sub : end_sub + 1]
    all_files = all_files[all_files["subject"].isin(subjects_to_keep)]

    subject_files = []
    # iterate over all files, subject by subject
    for i, files in tqdm(all_files.iterrows(), total=all_files.shape[0]):
        subject = files["subject"]

        run = files["run"]
        fmri_file = files["fmri"]

        print("loading fmri file...")
        fmri_file = image.load_img(fmri_file)
        print("resampling fmri file...")
        t0 = time.time()
        fmri_file = image.resample_to_img(fmri_file, mask_file)
        t1 = time.time()
        print(f"done resampling fmri file in {t1-t0}...")

        loss_events = pd.read_csv(
            files["loss"],
            sep="\t",
            header=None,
            index_col=None,
            names=["onset", "duration", "trial_type"],
        )
        loss_events["trial_type"] = ["loss"] * len(loss_events)
        win_events = pd.read_csv(
            files["win"],
            sep="\t",
            header=None,
            index_col=None,
            names=["onset", "duration", "trial_type"],
        )
        win_events["trial_type"] = ["win"] * len(win_events)
        events_df = pd.concat([loss_events, win_events])
        events_df.reset_index(drop=True, inplace=True)

        # skip subject if only one condition
        if len(events_df["trial_type"].unique()) < len(main_conditions):
            continue
        # skip subject if not enough trials
        if len(events_df) < 10:
            continue

        # drop nans or non-main trials, if any
        events_df = events_df.loc[
            events_df["trial_type"].isin(main_conditions)
        ]

        # Load the confounds
        movement = np.loadtxt(files["movement"])
        derivative = np.loadtxt(files["derivative"])
        confounds = np.hstack([movement, derivative])
        compcorr = image.high_variance_confounds(fmri_file)
        confounds = np.hstack([confounds, compcorr])
        confounds = pd.DataFrame(confounds)

        # Loop through the trials of interest and transform the DataFrame for LSS
        labels_runs_maps = Parallel(n_jobs=20, verbose=11)(
            delayed(get_trial_wise_maps)(
                fmri_file,
                mask_file,
                events_df,
                confounds,
                t_r,
                i_trial,
                subject,
                run,
                design_matrix_dir,
            )
            for i_trial in range(events_df.shape[0])
        )

        if i % 2 != 0:
            assert run == "LR"
            subject_files.extend(labels_runs_maps)
            # Unpack the labels and maps
            labels, runs, maps = zip(*subject_files)

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

            # runs for the beta maps
            run_file = os.path.join(three_mm_dir, f"{subject}_runs.csv")
            pd.Series(runs).to_csv(run_file, index=None, header=None)

            # reset subject files
            subject_files = []
        else:
            assert run == "RL"
            subject_files.extend(labels_runs_maps)
