from glob import glob
import os
import numpy as np
from nilearn import image
from nilearn.interfaces.fmriprep import load_confounds
from nilearn.glm.first_level import (
    make_first_level_design_matrix,
    FirstLevelModel,
)
from nilearn.plotting import (
    plot_design_matrix,
    plot_contrast_matrix,
    plot_stat_map,
)
from tqdm import tqdm
import pandas as pd
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from nilearn.datasets import load_mni152_gm_mask


def make_dmtx(events, fmri, main_conditions, confounds=None, t_r=2):
    from pandas import read_csv
    import nibabel as nib

    n_scans = nib.load(fmri).shape[3]
    frame_times = np.arange(n_scans) * t_r
    paradigm = read_csv(events, sep="\t")

    paradigm.rename(
        columns={
            "trial_type": "trial_type_old",
            "response_hand": "trial_type",
        },
        inplace=True,
    )

    trial_type = paradigm.trial_type.values

    print(trial_type)

    for condition in main_conditions:
        trial_idx = trial_type == condition
        n_trials = len(trial_type[trial_idx])
        trial_type[trial_idx] = [
            "%s_%02d" % (condition, i) for i in range(n_trials)
        ]
    paradigm.trial_type = trial_type

    print(trial_type)
    print(paradigm)

    dmtx = make_first_level_design_matrix(
        frame_times,
        events=paradigm,
        hrf_model="spm",
        add_regs=confounds,
    )

    plot_design_matrix(dmtx, output_file="design_matrix_trial.png")

    return dmtx


def trial_wise_zmaps_for_sub(
    df, subject, main_conditions, three_mm_dir, t_r=2
):
    df_ = df[df["subject"] == subject]
    print(df_)
    assert len(df_) == 1
    mask = load_mni152_gm_mask(resolution=3)
    for i, row in df_.iterrows():
        model = FirstLevelModel(
            mask_img=mask,
            t_r=t_r,
            high_pass=1.0 / 128,
            smoothing_fwhm=5,
        )
        confounds, _ = load_confounds(row["fmri"])
        dmtx = make_dmtx(
            row["events"], row["fmri"], main_conditions, confounds, t_r=t_r
        )
        model.fit(row["fmri"], design_matrices=[dmtx])
        all_maps = []
        for trial in tqdm(
            dmtx.columns, desc=f"{subject}", total=len(dmtx.columns)
        ):
            name = trial[:-3]
            if name in main_conditions:
                z = model.compute_contrast(
                    dmtx.columns == trial, output_type="z_score"
                )
                all_maps.append(z)
        print(subject, len(all_maps))
        if len(all_maps) < 2:
            print(f"no maps for {subject}")
            continue
        z = image.concat_imgs(all_maps)
        print("before resampling:", z.shape)
        z.to_filename(os.path.join(three_mm_dir, f"{subject}.nii.gz"))

        return os.path.join(three_mm_dir, f"{subject}.nii.gz")


def localiser_contrast(
    df,
    subject,
    smoothing_fwhm=5,
    include_confounds=True,
    t_r=2,
    high_pass=1.0 / 128,
):
    main_contrasts = ["right - left", "left - right"]
    df_ = df[df["subject"] == subject]
    print(df_)
    assert len(df_) == 1
    mask = load_mni152_gm_mask(resolution=3)
    for i, row in df_.iterrows():
        if high_pass is not None:
            model = FirstLevelModel(
                mask_img=mask,
                t_r=t_r,
                high_pass=high_pass,
                smoothing_fwhm=smoothing_fwhm,
            )
        else:
            model = FirstLevelModel(
                mask_img=mask,
                t_r=t_r,
                smoothing_fwhm=smoothing_fwhm,
            )
        events = pd.read_csv(row["events"], sep="\t")
        events.rename(
            columns={
                "trial_type": "trial_type_old",
                "response_hand": "trial_type",
            },
            inplace=True,
        )
        if include_confounds:
            confounds, _ = load_confounds(row["fmri"])
            model = model.fit(row["fmri"], events=events, confounds=confounds)
        else:
            model = model.fit(row["fmri"], events=events)
        design_matrix = model.design_matrices_[0]
        plot_design_matrix(design_matrix, output_file="design_matrix_run.png")
        contrast_matrix = np.eye(design_matrix.shape[1])
        contrasts = {
            column: contrast_matrix[i]
            for i, column in enumerate(design_matrix.columns)
        }
        contrasts["right - left"] = contrasts["right"] - contrasts["left"]
        contrasts["left - right"] = contrasts["left"] - contrasts["right"]
        # for key, values in contrasts.items():
        #     if key not in main_contrasts:
        #         continue
        # plot_contrast_matrix(values, design_matrix=design_matrix)
        # plt.suptitle(key)
        # plt.savefig(f"contrast_matrix_{key}.png")
        # plt.close()

        for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
            if contrast_id not in main_contrasts:
                continue
            z_map = model.compute_contrast(contrast_val, output_type="z_score")
            plot_stat_map(
                z_map,
                display_mode="z",
                threshold=3.0,
                title=contrast_id,
                output_file=f"contrast_{contrast_id}_smoothing_{smoothing_fwhm}_confounds_{include_confounds}_high_pass_{high_pass}.png",
            )
            z_map.to_filename(f"z_map_{subject}.nii.gz")


if __name__ == "__main__":
    ROOT = "/storage/store2/work/haggarwa/retreat_2023"
    aomic = os.path.join(ROOT, "data", "aomic")
    derivatives = os.path.join(aomic, "ds002785", "derivatives", "fmriprep")
    source_dir = os.path.join(aomic, "ds002785")

    # hand conditions
    main_conditions = [
        "right",
        "left",
    ]

    # voxels are already 3mm, 3mm x 3mm x 3.3mm to be exact
    # so we don't need to resample
    three_mm_dir = os.path.join(ROOT, "test_glm", "aomic_gstroop", "3mm")
    os.makedirs(three_mm_dir, exist_ok=True)

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
    cutoff = 1
    subjects = subjects[:cutoff]
    fmri = fmri[:cutoff]
    events = events[:cutoff]
    masks = masks[:cutoff]

    df = pd.DataFrame(
        {
            "subject": subjects,
            "fmri": fmri,
            "events": events,
            "mask": masks,
        }
    )

    n_jobs = 50
    map_paths = Parallel(n_jobs=n_jobs)(
        delayed(trial_wise_zmaps_for_sub)(
            df, subject, main_conditions, three_mm_dir
        )
        for subject in subjects
    )

    ### create event label files
    for event, subject in zip(events, subjects):
        eve = pd.read_csv(event, sep="\t")

        eve.rename(
            columns={
                "trial_type": "trial_type_old",
                "response_hand": "trial_type",
            },
            inplace=True,
        )

        eve = eve["trial_type"]
        eve = eve[eve.isin(main_conditions)]

        # if len(df) < 2:
        #     print(f"no labels for {subject}")
        #     continue

        eve.to_csv(
            os.path.join(three_mm_dir, f"{subject}_labels.csv"),
            index=False,
            header=False,
        )

    # do a glm for full run
    localiser_contrast(df, "sub-0001")
