# Decoding

So far this repo contains code for decoding trial conditions in [IBC RSVPLanguage](https://individual-brain-charting.github.io/docs/tasks.html#rsvplanguage) fMRI task from voxel-wise, trial-by-trial GLM parameter values.

## How to run

* [todo] Run `fetch_ibc.py` to download preprocessed IBC data from EBRAINS
* Run `separate_events_scripts.py` to get voxel-wise, trial-by-trial GLM parameter values (or activations).
* Run `decoder.py` to decode trial conditions from the trial-by-trial activations.