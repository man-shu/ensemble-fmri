# Ensemble Learning and Decoding

This repo contains code for decoding trial conditions in four different cohorts and tasks: BOLD5000 (Chang et al. 2019), Forrest (Hanke et al. 2015), Neuromod (Bellec and Boyle 2019), and RSVP (Pinho et al. 2020).

## Method

We use trial-by-trial GLM parameter maps from each of these tasks. Each of these datasets has about 4-6 classes of stimuli to be decoded and hence has similar decoding complexity. We also compare the two decoding settings across two different feature spaces - full voxel space and 1024 modes of the DiFuMo atlas (Dadi et al. 2020) and use two different classifiers for decoding - linear SVC and random forest (Pedregosa et al. 2011).

Within each cohort, we vary the size of the training set in increments of 10% of the samples available for each subject and always test the trained model on 10% of the samples.

In a conventional decoding setting, a classifier is trained to learn the mapping between stimuli labels and features in the voxel space. 

In the stacking approach we train a classifier to learn the mapping between true stimuli labels and the stimuli predictions from classifiers pre-trained on other subjects’ voxel-space features. This converts a high-dimensional problem (where the number of features is the number of voxels) to a low-dimensional one (where the number of features is one minus the number of subjects in the cohort).