# Natural Scenes Dataset on drago

* Please submit this form before using the data: https://forms.gle/eT4jHxaWwYUDEf2i9
* NSD data manual: https://cvnlab.slite.page/p/CT9Fwl4_hc/NSD-Data-Manual

## 1mm
* Contains only the single-trial beta values in MNI space, described here: https://cvnlab.slite.page/p/6CusMRYfk0/Functional-data-NSD#aa25c288
* downloaded using:
    ```bash
    aws s3 sync --dryrun --exclude "*" --include "*MNI/betas_fithrf/*"  s3://natural-scenes-dataset/nsddata_betas/ppdata/ .
    ```
* Remove `--dryrun` to actually download the data.
* This downloads `subj0{1-8}/MNI/betas_fithrf/betas_session*.nii.gz` and `subj0{1-8}/MNI/betas_fithrf/valid_session*.nii.gz` files.
* I restructured all that under `1mm/subj0{1-8}/betas_session*.nii.gz` and `1mm/subj0{1-8}/valid_session*.nii.gz`.
* These files had been "liberally masked" as mentioned here: https://cvnlab.slite.page/p/QtQPzl1xnH/FAQ#cf1f430a. Hence needed masking to keep only grey-matter voxels.

## 3mm
* Contains downsampled and masked version of the 1mm files.
* Only the `betas_session*.nii.gz` were resampled and masked.
* Created using `scripts/resample_mask.py`.

## info/nsd_*
* These are the supporting information files for the NSD experiment.
* Described here: https://cvnlab.slite.page/p/NKalgWd__F/Experiments#bfb23b56
* Downloaded using:
    ```bash
    aws s3 cp s3://natural-scenes-dataset/nsddata/experiments/nsd/nsd_stim_info_merged.pkl .
    ```
    or 

    ```bash
    aws s3 cp s3://natural-scenes-dataset/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5 .
    ```

## info/coco_annotations/
* Contains the COCO 2017 Train/Val annotation files downloaded from: https://cocodataset.org/#download and http://images.cocodataset.org/annotations/annotations_trainval2017.zip
* The `merged_instances_train_val_2017.json` was created by merging `instances_train_2017.json` and `instances_val_2017.json` using `scripts/merge.py` (from https://github.com/mohamadmansourX/Merge_COCO_FILES/blob/master/merge.py)
* The merged json file above was then used to get the COCO category and supercategory labels stored in `coco_categories.pkl` using `scripts/get_coco_labels.py`.
* Most of the COCO pictures have more than one supercategory/category label associated with them . All of them are stored in`supercats` or `cats` column in `coco_categories.pkl` dataframe.
* I saved the label that covers maximum area (areas saved in `areas` column) in the picture in separate `supercat` and `cat` columns in `coco_categories.pkl` dataframe.

## curated_3mm
**NOTE: please refrain from making any changes in this subdirectory**
* Contains 3mm images curated specifically for my application which was to decode stimulus labels (COCO picture supercategories) from the beta values.
* Not all NSD subjects could perform all the planned 30000 trials, see: https://cvnlab.slite.page/p/h_T_2Djeid/Technical-notes#65715439
* So the curation process only keeps first 22500 samples per subject, removes samples where COCO pictures with unknown categories were shown and finally removes some 'person' supercategory samples to have equal number of samples per subject.
* Done by `scripts/curate.py`.
* The result is `sub-0{1-8}.nii.gz` 4d images each with shape (62, 74, 62, 22274)
* Each 4d image has a corresponding `sub-0{1-8}_labels.npy` file which contains COCO supercategory labels associated with each COCO picture shown to NSD subjects.
* Each 4d image also has a corresponding `sub-0{1-8}_runs.npy` which contains run number associated with each COCO picture shown to NSD subjects.
