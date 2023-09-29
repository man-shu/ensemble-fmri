# Decoding

So far this repo contains code for decoding trial conditions in [IBC RSVPLanguage](https://individual-brain-charting.github.io/docs/tasks.html#rsvplanguage) fMRI task from voxel-wise, trial-by-trial GLM parameter values.

## How to run

* Run `fetch_ibc.py` to download preprocessed IBC data from EBRAINS
* Run `separate_events_script.py` to get voxel-wise, trial-by-trial GLM parameter values (or activations).
* Run `decoder.py` to decode trial conditions from the trial-by-trial activations. This is done in two ways:
    1. Across subjects: Here we train a Linear SVM classifier on 12 (out of 13) subjects and test on the left-out subject
    2. Within each subject: Here we train on 5 (out 6) runs and test on left-out run for each subject