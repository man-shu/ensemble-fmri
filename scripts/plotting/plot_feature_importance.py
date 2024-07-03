"""
This script projects the feature importance maps from DiFuMo back to the voxel 
space and plots them on the surface and glass brain.

"""

import os
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
from nilearn import maskers, datasets, plotting
from nilearn import image
from joblib import load, Parallel, delayed

sns.set_context("talk", font_scale=1.2)
DATA_ROOT = "."


def fix_names(df):
    if "setting" in list(df.columns):
        df["Setting"] = df["setting"].apply(
            lambda x: "Conventional" if x == "conventional" else "Ensemble"
        )
    elif "Setting" in list(df.columns):
        df["Setting"] = df["Setting"].apply(
            lambda x: "Conventional" if x == "conventional" else "Ensemble"
        )
    if "feature" in list(df.columns):
        df["Feature"] = df["feature"].apply(
            lambda x: "Voxels" if x == "voxels" else "DiFuMo"
        )
    elif "Feature" in list(df.columns):
        df["Feature"] = df["Feature"].apply(
            lambda x: "Voxels" if x == "voxels" else "DiFuMo"
        )
    return df


def get_subs(dataset, feature, classifier):
    featimp_dir = os.path.join(results_root, f"{dataset}_{feature}_featimp")
    pkls = glob(os.path.join(featimp_dir, f"*{classifier}*.pkl"))
    subs = [os.path.basename(pkl).split("_")[-1].split(".")[0] for pkl in pkls]

    return subs


def load_sub(
    dataset,
    feature,
    classifier,
    subject,
    results_root,
):
    featimp_dir = os.path.join(results_root, f"{dataset}_{feature}_featimp")
    pkl = glob(os.path.join(featimp_dir, f"*{classifier}*{subject}.pkl"))[0]
    print(pkl)
    sub_df = load(pkl)

    return sub_df


def importance_on_voxels(importances, sample_idx, masker):
    importance = importances[:, sample_idx]
    importance_on_voxels = masker.inverse_transform(importance)

    return importance_on_voxels


def compute_imp_std(pred_scores):
    weights = np.array([el.shape[-1] for el in pred_scores])
    # Compute the mean of each fold over the number of observations
    pred_mean = np.array([np.mean(el.copy(), axis=-1) for el in pred_scores])

    # Weighted average
    imp = np.average(pred_mean, axis=0, weights=weights)

    # Compute the standard deviation of each fold
    # over the number of observations
    pred_std = np.array(
        [
            np.mean(
                (el - imp[..., np.newaxis]) ** 2,
                axis=-1,
            )
            for el in pred_scores
        ]
    )
    std = np.sqrt(
        np.average(pred_std, axis=0, weights=weights) / (np.sum(weights) - 1)
    )
    return (imp, std)


if __name__ == "__main__":
    N_JOBS = 5

    data_dir = os.path.join(DATA_ROOT, "data", "feat_imp")
    results_root = os.path.join(DATA_ROOT, "results")
    out_dir = os.path.join(DATA_ROOT, "plots_copy", "thresholded_featimp")
    os.makedirs(out_dir, exist_ok=True)

    # datasets
    datas = [
        "neuromod",
        "aomic_anticipation",
        "forrest",
        "rsvp",
    ]

    # Camelized dataset names
    fixed_datasets = {
        "neuromod": "Neuromod",
        "forrest": "Forrest",
        "bold5000": "BOLD5000",
        "rsvp": "RSVP-IBC",
        # "nsd": "NSD",
        "aomic_anticipation": "AOMIC",
    }
    n_samples = [50, 61, 175, 360]
    n_subs = [4, 203, 10, 13]

    features = ["difumo"]
    classifiers = ["RandomForest"]

    settings = ["conventional", "ensemble"]

    for dataset_i, dataset in enumerate(datas):
        for feature in features:
            for classifier in classifiers:
                subs = get_subs(dataset, feature, classifier)
                print(subs, dataset)
                for sub in subs:
                    z_map_path = os.path.join(
                        data_dir,
                        f"{dataset}_{feature}_{classifier}_{sub}_featimp_voxels_z.nii.gz",
                    )
                    if os.path.exists(z_map_path):
                        z_map = image.load_img(z_map_path)
                    else:
                        atlas = datasets.fetch_atlas_difumo(
                            dimension=1024,
                            resolution_mm=3,
                            data_dir=DATA_ROOT,
                            legacy_format=False,
                        )
                        atlas["name"] = "difumo"

                        mask = datasets.load_mni152_gm_mask(resolution=3)

                        masker = maskers.NiftiMapsMasker(
                            maps_img=atlas["maps"],
                            # mask_img=mask,
                            verbose=11,
                            n_jobs=20,
                            memory="difumo_to_voxels_cache",
                        ).fit()
                        voxel_importance = {"imgs": []}
                        sub_df = load_sub(
                            dataset, feature, classifier, sub, results_root
                        )
                        importances = [
                            sub_df["all_scores_fold1"],
                            sub_df["all_scores_fold2"],
                        ]

                        for i, imp in enumerate(importances):
                            n_samples = imp.shape[1]

                            voxel_imp = Parallel(
                                n_jobs=N_JOBS,
                                verbose=11,
                            )(
                                delayed(importance_on_voxels)(
                                    imp, sample_i, masker
                                )
                                for sample_i in range(n_samples)
                            )
                            voxel_importance["imgs"].append(
                                image.concat_imgs(voxel_imp)
                            )

                        voxel_imp_arrays = []
                        for i, voxel_imp in enumerate(
                            voxel_importance["imgs"]
                        ):
                            voxel_imp.to_filename(
                                os.path.join(
                                    data_dir,
                                    f"{dataset}_{feature}_{classifier}_{sub}_featimp_voxels_fold{i+1}.nii.gz",
                                )
                            )
                            voxel_imp_arrays.append(voxel_imp.get_fdata())

                        mean, std = compute_imp_std(voxel_imp_arrays)
                        z_map = mean / std
                        np.seterr(divide="ignore", invalid="ignore")
                        z_map = image.new_img_like(
                            voxel_importance["imgs"][0], z_map
                        )
                        z_map.to_filename(
                            os.path.join(
                                data_dir,
                                f"{dataset}_{feature}_{classifier}_{sub}_featimp_voxels_z.nii.gz",
                            )
                        )
                    # plotting.plot_img_on_surf(
                    #     z_map,
                    #     title=f"{fixed_datasets[dataset]}",
                    #     fsaverage="fsaverage",
                    #     views=["lateral"],
                    # )
                    # plt.savefig(
                    #     os.path.join(
                    #         out_dir,
                    #         f"{dataset}_{feature}_{classifier}_{sub}_featimp_voxels_z.png",
                    #     ),
                    #     bbox_inches="tight",
                    # )
                    # plt.savefig(
                    #     os.path.join(
                    #         out_dir,
                    #         f"{dataset}_{feature}_{classifier}_{sub}_featimp_voxels_z.svg",
                    #     ),
                    #     bbox_inches="tight",
                    # )
                    # plt.close()

                    plotting.plot_glass_brain(
                        z_map,
                        title=f"{fixed_datasets[dataset]}",
                        plot_abs=False,
                        cmap="coolwarm",
                        colorbar=True,
                        threshold=2.054,
                        symmetric_cbar=False,
                        vmin=0,
                    )
                    plt.savefig(
                        os.path.join(
                            out_dir,
                            f"{dataset}_{feature}_{classifier}_{sub}_featimp_voxels_z_glass.png",
                        ),
                        bbox_inches="tight",
                    )
                    plt.savefig(
                        os.path.join(
                            out_dir,
                            f"{dataset}_{feature}_{classifier}_{sub}_featimp_voxels_z_glass.svg",
                        ),
                        bbox_inches="tight",
                    )
                    plt.close()
