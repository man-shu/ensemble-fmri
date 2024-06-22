# Ensemble Learning and Decoding

## How to run

### Clone the repository

```bash
git clone git@github.com:man-shu/ensemble-fmri.git
```

### Download the data

Download downsampled 3mm event-wise GLM effect-size maps of fMRI datasets.

```bash
cd ensemble-fmri
wget https://zenodo.org/records/12204275/files/data.zip
unzip data.zip -d data
```

### Install the dependencies

Create a new conda environment with the dependencies.

```bash
conda env create -f environment_nogpu.yml
conda activate ensemble_fmri_nopgpu
```

### Run the experiments

To generate numbers plotted in Fig 2 and Fig 3 (over varying training sizes) in the paper:

* with DiFuMo features, run the following command:

    ```bash
    python scripts/decode_vary_train_size/difumo.py
    ```

* with voxel-wise features, run the following command:

    ```bash
    python scripts/decode_vary_train_size/wholebrain.py
    ```

To generate numbers plotted in Fig 4 (over varying number of subjects in the ensemble) in the paper:

* with DiFuMo features, run the following command:

    ```bash
    python scripts/decode_vary_subjects/vary_n_subs_difumo.py
    ```

* with voxel-wise features, run the following command:

    ```bash
    python scripts/decode_vary_subjects/vary_n_subs_wholebrain.py
    ```


## Abstract

Decoding cognitive states from functional magnetic resonance imaging is central to understanding the functional organization of the brain. Within-subject decoding avoids between-subject correspondence problems but requires large sample sizes to make accurate predictions; obtaining such large sample sizes is both challenging and expensive. Here, we investigate an ensemble approach to decoding that combines the classifiers trained on data from other subjects to decode cognitive states in a new subject. We compare it with the conventional decoding approach on five different datasets and cognitive tasks. We find that it outperforms the conventional approach by up to 20\% in accuracy, especially for datasets with limited per-subject data. The ensemble approach is particularly advantageous when the classifier is trained in voxel space. Furthermore, a Multi-layer Perceptron turns out to be a good default choice as an ensemble method. These results show that the pre-training strategy reduces the need for large per-subject data.


## Results

#### The ensemble approach outperforms conventional decoding
![Average decoding accuracy: Each plot represents a different dataset (along columns). The average decoding accuracy is plotted along the x-axis. The averages are across all training sizes, subjects and 20 cross-validation splits. The error bars represent a 95\% confidence interval of the bootstrap distribution. The horizontal line represents the chance level of accuracy.](plots/bench_balanced_accuracy.png "Average decoding accuracy")

#### It is beneficial in scarce data scenarios
![Gain in decoding accuracy when varying the number of training samples per class: Each plot represents a different dataset (along columns). The y-axis shows the average percent gain in decoding accuracy (accuracy of ensemble - accuracy of conventional) across all subjects and 20 cross-validation splits. On the x-axis, training size is reported as the number of samples per class in each cross-validation split. The confidence intervals represent a 95% confidence interval of bootstrap distribution. The horizontal line represents no average gain in accuracy and the vertical line, 10 samples per class.](plots/gains_v_samples_per_class_balanced_accuracy.png "Gain in decoding accuracy when varying the number of training samples per class")

#### The gains increase with increasing number of subjects in the ensemble
![Gain in decoding accuracy over a varying number of subjects in the ensemble: Each plot represents a different dataset (along columns). The x-axis represents the number of subjects used in the ensemble method. The y-axis represents the average percent gain in decoding accuracy (accuracy of ensemble - accuracy of conventional) across all training sizes and 5 cross-validation splits. The confidence intervals represent a 95% interval of bootstrap distribution. The horizontal line represents no average gain in accuracy and the vertical line at 10 subjects in the ensemble.](plots/varysubs_vs_gain.png "Gain in decoding accuracy over a varying number of subjects in the ensemble")

#### Extracting voxel-wise feature importance scores is still possible
![](plots/supp/rsvp_difumo_RandomForest_sub-04_featimp_voxels_z_glass.png "Broca's area is important for decoding conditions in the RSVP language task")


## More details

### Install conda

If you don't have conda installed, you can install it from [here](https://docs.conda.io/en/latest/miniconda.html).