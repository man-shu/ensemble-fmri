# %% [markdown]
# # Brain decoding with GCN
# 
# 
# ## Brain graph representation
# 
# Graph signal processing is a new tool to model brain organization and function. The brain is composed of several Region of Interests(ROIs). Brain graphs provide an efficient way for modeling the human brain connectome, by associating nodes to the brain regions, and defining edges via anatomical or functional connections. These ROIs are connected to some regions of interests with the highest connectivity.
# <br/><br/>
# 
# <img src="Brain_connectivity_graph.png" width=545 height=194 />
# 
# Representation of Brain connectivity by graph theory. 
# Image source:https://atcold.github.io/pytorch-Deep-Learning/en/week13/13-1/ 
# 
# ## Graph Convolution Network (GCN)
# <br/><br/>
# <img src="GCN_pipeline_main2022.png" width=680 height=335 />
# 
# 
# Schematic view of brain decoding using graph convolution network. Model is adapted from Zhang and colleagues (2021). 
# **a)** Bold time series are used to construct the brain graph by associating nodes to predefined brain regions (parcels) and indicating edges between each pair of brain regions based on the strength of their connections. Then, both brain graph and time-series matrix are imported into the graph convolutional network 
# **b)** The decoding model consists of three graph convolutional layers with 32 ChebNet graph filters at each layer,  followed by a global average pooling layer, two fully connected layers (MLP, consisting of 256-128 units) and softmax function. This pipeline generates task-specific representations of recorded brain activities and predicts the corresponding cognitive states.
# 
# ## Getting the data
# 
# We are going to download the dataset from Haxby and colleagues (2001) {cite:p}`Haxby2001-vt`. You can check {ref}`haxby-dataset` section for more details on that dataset. Here we are going to quickly download it, and prepare it for machine learning applications with a set of predictive variable, the brain time series, and a dependent variable, the annotation on cognition.

# %%
import os
import warnings
warnings.filterwarnings(action='once')
from nilearn.maskers import NiftiMasker

from nilearn import datasets

# We are fetching the data for subject 4
data_dir = os.path.join('..', 'data')
sub_no = 4
haxby_dataset = datasets.fetch_haxby(subjects=[sub_no], fetch_stimuli=True, data_dir=data_dir)
func_file = haxby_dataset.func[0]

# Standardizing
mask_vt_file = haxby_dataset.mask_vt[0]
masker = NiftiMasker(mask_img=mask_vt_file, standardize=True)

# cognitive annotations
import pandas as pd
behavioral = pd.read_csv(haxby_dataset.session_target[0], delimiter=' ')
X = masker.fit_transform(func_file)
y = behavioral['labels']

# %% [markdown]
# Let's check the shape of X and y and the cognitive annotations of this data sample.

# %%
categories = y.unique()
print(categories)
print('y:', y.shape)
print('X:', X.shape)

# %% [markdown]
# So we have 1452 time points in the imaging data, and for each time point we have recordings of fMRI activity across 675 brain regions.
# 
# ## Create brain graph for GCN
# 
# A key component of GCN is brain graph.
# Brain graph provides a network representation of brain organization by associating nodes to brain regions and defining edges via anatomical or functional connections.
# After generating time series, we will firstly use the nilearn function to geneate a correlation based functional connectome.
# 
# ```{admonition} Basic of graph laplacian and graph convolutional networks.
# :class: tip
# To explore the basics of `graph laplacian` and `graph convolutional networks` and how to apply these tools to neuroimging data check the tutorial from MAIN 2019 conference presented by Dr. Zhang.
# 
# [GCN_tutorial_slides:](https://drive.google.com/file/d/1Gu28WcHXlwjXQSSmqZZwIcESHff_j-J4/view?usp=sharing)  
# [Github repo:](https://github.com/zhangyu2ustc/gcn_tutorial_test.git)
# [Binder projects:](https://mybinder.org/v2/gh/zhangyu2ustc/gcn_tutorial_test/master?filepath=notebooks%2F) 
# ```

# %%
import warnings
warnings.filterwarnings(action='once')

import nilearn.connectome

# Estimating connectomes and save for pytorch to load
corr_measure = nilearn.connectome.ConnectivityMeasure(kind="correlation")
conn = corr_measure.fit_transform([X])[0]

n_regions_extracted = X.shape[-1]
title = 'Correlation between %d regions' % n_regions_extracted

print('Correlation matrix shape:',conn.shape)

# First plot the matrix
from nilearn import plotting
display = plotting.plot_matrix(conn, vmax=1, vmin=-1,
                               colorbar=True, title=title)

# %% [markdown]
# The next step is to construct the brain graph for GCN.
# 
# __k-Nearest Neighbours(KNN) graph__ for the group average connectome will be built based on the connectivity-matrix.
# 
# Each node is only connected to *k* conn = corr_measure.fit_transform([X])[0]
# other neighbouring nodes.
# For the purpose of demostration, we constrain the graph to from clusters with __8__ neighbouring nodes with the strongest connectivity.
# 
# For more details you please check out __*src/graph_construction.py*__ script.

# %%
from graph_construction import make_group_graph

# make a graph for the subject
graph = make_group_graph([conn], self_loops=False, k=8, symmetric=True)

# %% [markdown]
# ## Preparing the dataset for model training
# 
# The trials for different object categories are scattered in the experiment. 
# Firstly we will concatenated the volumes of the same category together.

# %%
# generate data
import pandas as pd
import numpy as np

# cancatenate the same type of trials
concat_bold = {}
for label in categories:
    cur_label_index = y.index[y == label].tolist()
    curr_bold_seg = X[cur_label_index]    
    concat_bold[label] = curr_bold_seg

# %% [markdown]
# We split the data by the time window size that we wish to use to caputre the temporal dynamic.
# Different lengths for our input data can be selected. 
# In this example we will continue with __*window_length = 1*__, which means each input file will have a length equal to just one Repetition Time (TR).
# The splitted timeseries are saved as individual files (in the format of `<category>_seg_<serialnumber>.npy`), 
# the file names and the associated label are stored in the same directory,
# under a file named `label.csv`.

# %%
# split the data by time window size and save to file
window_length = 1
dic_labels = {name: i for i, name in enumerate(categories)}

# set output paths
split_path = os.path.join(data_dir, 'haxby_split_win/')
if not os.path.exists(split_path):
    os.makedirs(split_path)
out_file = os.path.join(split_path, '{}_{:04d}.npy')
out_csv = os.path.join(split_path, 'labels.csv')

label_df = []
for label, ts_data in concat_bold.items():
    ts_duration = len(ts_data)
    ts_filename = f"{label}_seg"
    valid_label = dic_labels[label]

    # Split the timeseries
    rem = ts_duration % window_length
    n_splits = int(np.floor(ts_duration / window_length))

    ts_data = ts_data[:(ts_duration - rem), :]   

    for j, split_ts in enumerate(np.split(ts_data, n_splits)):
        ts_output_file_name = out_file.format(ts_filename, j)

        split_ts = np.swapaxes(split_ts, 0, 1)
        np.save(ts_output_file_name, split_ts)

        curr_label = {'label': valid_label, 'filename': os.path.basename(ts_output_file_name)}
        label_df.append(curr_label)
label_df = pd.DataFrame(label_df)
label_df.to_csv(out_csv, index=False)  

# %% [markdown]
# Now we use a customised `pytorch` dataset generator class `TimeWindowsDataset` to split the data into training, 
# validation, and testing sets for model selection.
# 
# The dataset generator defaults isolates 20% of the data as the validation set, and 10% as testing set.
# For more details of customising a dataset, please see `src/gcn_windows_dataset.py` and the 
# official [`pytorch` documentation](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files).

# %%
# split dataset
from gcn_windows_dataset import TimeWindowsDataset

random_seed = 0

train_dataset = TimeWindowsDataset(
    data_dir=split_path, 
    partition="train", 
    random_seed=random_seed, 
    pin_memory=True, 
    normalize=True,
    shuffle=True)

valid_dataset = TimeWindowsDataset(
    data_dir=split_path, 
    partition="valid", 
    random_seed=random_seed, 
    pin_memory=True, 
    normalize=True,
    shuffle=True)

test_dataset = TimeWindowsDataset(
    data_dir=split_path, 
    partition="test", 
    random_seed=random_seed, 
    pin_memory=True, 
    normalize=True,
    shuffle=True)

print("train dataset: {}".format(train_dataset))
print("valid dataset: {}".format(valid_dataset))
print("test dataset: {}".format(test_dataset))

# %% [markdown]
# Once the datasets are created, we can use the pytorch [data loader](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#preparing-your-data-for-training-with-dataloaders) to iterate through the data during the model selection process.
# The __batch size__ defines the number of samples that will be propagated through the neural network.
# We are separating the dataset into 10 time windows per batch.

# %%
import torch
from torch.utils.data import DataLoader

batch_size = 10

torch.manual_seed(random_seed)
train_generator = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_generator = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
test_generator = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
train_features, train_labels = next(iter(train_generator))
print(f"Feature batch shape: {train_features.size()}; mean {torch.mean(train_features)}")
print(f"Labels batch shape: {train_labels.size()}; mean {torch.mean(torch.Tensor.float(train_labels))}")

# %% [markdown]
# ## Generating a GCN model 
# 
# We have created a GCN of the following property:
# - __3__ graph convolutional layers
# - __32 graph filters__  at each layer
# - followed by a __global average pooling__ layer
# - __2 fully connected__ layers

# %%
from gcn_model import GCN

gcn = GCN(graph.edge_index, 
          graph.edge_attr, 
          n_roi=X.shape[1],
          batch_size=batch_size,
          n_timepoints=window_length, 
          n_classes=len(categories))
gcn

# %% [markdown]
# ## Train and evaluating the model
# 
# We will use a procedure called backpropagation to train the model.
# When we training the model with the first batch of data, the accuarcy and loss will be pretty poor.
# Backpropagation is an algorithm to update the model based on the rate of loss. 
# Iterating through each batch, the model will be updated and reduce the loss.
# 
# Function `training_loop` performs backpropagation through pytorch. 
# One can use their own choice of optimizer for backpropagation and estimator for loss.
# 
# After one round of training, we use the validation dataset to calculate the average accuracy and loss with function `valid_test_loop`. 
# These metrics will serve as the reference for model performance of this round of training.

# %%
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)    

    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss, current = loss.item(), batch * dataloader.batch_size

        correct = (pred.argmax(1) == y).type(torch.float).sum().item()
        correct /= X.shape[0]
        if (batch % 10 == 0) or (current == size):
            print(f"#{batch:>5};\ttrain_loss: {loss:>0.3f};\ttrain_accuracy:{(100*correct):>5.1f}%\t\t[{current:>5d}/{size:>5d}]")

        
def valid_test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model.forward(X)
            loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    loss /= size
    correct /= size

    return loss, correct

# %% [markdown]
# This whole procedure described above is called an __epoch__.
# We will repeat the process for 25 epochs.
# Here the choice of loss function is `CrossEntropyLoss` and the optimizer to update the model is `Adam`.

# %%
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(gcn.parameters(), lr=1e-4, weight_decay=5e-4)

epochs = 25
for t in range(epochs):
    print(f"Epoch {t+1}/{epochs}\n-------------------------------")
    train_loop(train_generator, gcn, loss_fn, optimizer)
    loss, correct = valid_test_loop(valid_generator, gcn, loss_fn)
    print(f"Valid metrics:\n\t avg_loss: {loss:>8f};\t avg_accuracy: {(100*correct):>0.1f}%")

# %% [markdown]
# After training the model for 25 epochs, we use the untouched test data to evaluate the model and conclude the results of training.

# %%
# results
loss, correct = valid_test_loop(test_generator, gcn, loss_fn)
print(f"Test metrics:\n\t avg_loss: {loss:>f};\t avg_accuracy: {(100*correct):>0.1f}%")

# %% [markdown]
# The performance is good but how could we still improve it?
# 
# ## Exercises
#  * Try out different time window sizes, batch size for the dataset,
#  * Try different brain graph construction methods.
#  * Try use different loss function or optimizer function.
#  * **Hard**: Treat the parameters you changed, such as time window size and batch size, as parameters of part of the model training.
#  * **Hard**: Try extracting regions from network components using dictionary learning for estimating brain networks.

# %% [markdown]
# ## References
# 
# ```{bibliography}
# :filter: docname in docnames
# ```


