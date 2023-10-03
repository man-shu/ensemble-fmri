# Decoding

So far this repo contains code for decoding trial conditions in [IBC RSVPLanguage](https://individual-brain-charting.github.io/docs/tasks.html#rsvplanguage) fMRI task from voxel-wise, trial-by-trial GLM parameter values.

## How to run

* Run `fetch_ibc.py` to download preprocessed IBC data from EBRAINS
* Run `separate_events_script.py` to get voxel-wise, trial-by-trial GLM parameter values (or activations).
* Run `decoder.py` to decode trial conditions from the trial-by-trial activations. This is done in three ways:
    1. Across subjects: Here we train a Linear SVM classifier on 12 (out of 13) subjects and test on the left-out subject
    2. Within each subject: Here we train on 5 (out 6) runs and test on left-out run for each subject
    3. Within each run for each subject: Here we train on 80% of the trials in a run and test on remaining 20% within each run for each subject. (Accuracy scores are potentially biased due to the autocorrelated signal within runs)
* Run `stacking.py` to do the decoding via a stacking approach. Here we stack predictions from pre-trained classifiers and cross-validate a final classifier to map those predictions to the classes of stimuli. This is done in 2 ways:
    1. Across subjects: Here we pre-train different classifiers on each of 12 (out of 13) subjects and cross-validate with leave-one-run-out scheme on the remaining subject
    2. Within subjects: Here we pre-train on 5 (out of 6) runs and cross-validate with a shuffled-stratfied scheme with 80-20 train-test ratio. This is done for each subject separately