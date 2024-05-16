from glob import glob
import os
import numpy as np
from nilearn import image
from nilearn.interfaces.fmriprep import load_confounds
from tqdm import tqdm
import pandas as pd
from joblib import Parallel, delayed
import nibabel as nib
import ibc_public.utils_data
import ibc_public
import warnings
from nilearn.datasets import fetch_atlas_difumo
from nilearn.maskers import NiftiMapsMasker
import matplotlib.pyplot as plt
from gcn_windows_dataset import TimeWindowsDataset
import torch
from torch.utils.data import DataLoader
from gcn_model import GCN
import nilearn.connectome
from graph_construction import make_group_graph

warnings.filterwarnings("ignore")

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


if __name__ == "__main__":
    ROOT = "/storage/store2/work/haggarwa/retreat_2023"
    ibc = os.path.join(ROOT, "data", "ibc")
    derivatives = os.path.join(ibc, "derivatives")

    # obtain a grey matter mask
    _package_directory = os.path.dirname(
        os.path.abspath(ibc_public.utils_data.__file__)
    )
    mask_gm = os.path.join(
        _package_directory, "../ibc_data", "gm_mask_3mm.nii.gz"
    )

    # sessions + subject data
    session_file = os.path.join(
        _package_directory, "../ibc_data", "sessions.csv"
    )
    subject_session = ibc_public.utils_data.get_subject_session(
        "rsvp-language"
    )

    split_ts = os.path.join(ROOT, "data", "rsvp", "split_time_series_difumo")
    os.makedirs(split_ts, exist_ok=True)

    time_series_dir = os.path.join(ROOT, "data", "rsvp", "plots_time_series")
    os.makedirs(time_series_dir, exist_ok=True)

    # hand conditions
    main_conditions = [
        "complex",
        "simple",
        "consonant_strings",
        "word_list",
        "jabberwocky",
        "pseudoword_list",
    ]

    # TR
    t_r = 2.0

    # fetch difumo
    atlas = fetch_atlas_difumo(
        dimension=1024,
        resolution_mm=3,
        data_dir=ROOT,
        legacy_format=False,
    )
    atlas_name = "difumo"
    masker = NiftiMapsMasker(
        maps_img=atlas.maps,
        mask_img=mask_gm,
        standardize="zscore_sample",
        memory="nilearn_cache",
        verbose=5,
    )
    
    event_count = 0
    encoded_labels = {name: i for i, name in enumerate(main_conditions)}
    labels_df = []
    X = []

    for subject, session in tqdm(subject_session):
        data_path = os.path.join(derivatives, subject, session, "func")
        fmri = sorted(
            glob(os.path.join(data_path, "w*RSVPLanguage*_bold.nii.gz"))
        )
        confounds = sorted(
            glob(os.path.join(data_path, "rp_*RSVPLanguage*.txt"))
        )
        events = sorted(
            glob(os.path.join(data_path, "*RSVPLanguage*_events.tsv"))
        )

        for fmri_file, confound_file, events_file in zip(fmri, confounds, events):
            confound_df = np.loadtxt(confound_file)
            events_df = pd.read_csv(events_file, sep="\t")

            run = fmri_file.split("/")[-1].split("_")[-2]

            # downsample fmri file
            fmri_file = image.resample_img(
                        fmri_file, target_affine=np.diag((3, 3, 3))
                    )

            # get region-wise time series
            time_series = masker.fit_transform(fmri_file, confounds=confound_df)
            print("Time series shape: ", time_series.shape)

            complexs = [
                "complex_sentence_objclef",
                "complex_sentence_objrel",
                "complex_sentence_subjrel",
            ]
            simples = [
                "simple_sentence_adj",
                "simple_sentence_coord",
                "simple_sentence_cvp",
            ]
            for complex_ in complexs:
                events_df = events_df.replace(complex_, "complex")
            for simple_ in simples:
                events_df = events_df.replace(simple_, "simple")

            # only keep main conditions
            events_df = events_df.loc[
                events_df["trial_type"].isin(main_conditions)
            ]
            events_df.reset_index(inplace=True, drop=True)

            # plot time series
            plt.figure(figsize=(20, 5))
            plt.plot([*range(0, time_series.shape[0]*int(t_r), int(t_r))], time_series[:,0])
            plt.scatter([*range(0, time_series.shape[0]*int(t_r), int(t_r))], time_series[:,0], s=1, color="k")
            onsets = events_df["onset"].to_list()
            durations = events_df["duration"].to_list()

            for onset, duration in zip(onsets, durations):
                plt.fill_between([onset, onset+duration], time_series[:,0].min(), time_series[:,0].max(), color="r", alpha=0.5)
            plt.savefig(os.path.join(time_series_dir, f"{subject}_{run}.png"), bbox_inches="tight", dpi=600)
            plt.close()

            # keep time points that lie within the event durations
            for row in events_df.iterrows():
                onset = row[1]["onset"]
                duration = row[1]["duration"]
                label = row[1]["trial_type"]
                time_points = time_series[(np.arange(0, time_series.shape[0]) * t_r >= onset) & (np.arange(0, time_series.shape[0]) * t_r <= onset + duration)]
                # save time points
                n_points = time_points.shape[0]
                for point in range(n_points):
                    time_point = time_points[point].T
                    time_point = np.expand_dims(time_point, axis=1)
                    filename = f"{subject}_{run}_{label}_{point}_{event_count}.npy"
                    np.save(os.path.join(split_ts, filename), time_point)
                    # save label
                    labels_df.append({"label": encoded_labels[label], "filename": filename})
                    # save time points
                    X.append(time_points[point])
                event_count += 1

    # labels dataframe
    labels_df = pd.DataFrame(labels_df)
    labels_df.to_csv(os.path.join(split_ts, "labels.csv"), index=False)

    # X and y
    X = np.array(X)
    y = labels_df["label"].values
    print(main_conditions)
    print('y:', y.shape)
    print('X:', X.shape)

    # Estimating connectomes and save for pytorch to load
    corr_measure = nilearn.connectome.ConnectivityMeasure(kind="correlation")
    conn = corr_measure.fit_transform([X])[0]

    n_regions_extracted = X.shape[-1]
    title = 'Correlation between %d regions' % n_regions_extracted

    print('Correlation matrix shape:',conn.shape)

    # make a graph for the subject
    graph = make_group_graph([conn], self_loops=False, k=8, symmetric=True)

    random_seed = 0

    train_dataset = TimeWindowsDataset(
        data_dir=split_ts, 
        partition="train", 
        random_seed=random_seed, 
        pin_memory=True, 
        normalize=True,
        shuffle=True)

    valid_dataset = TimeWindowsDataset(
        data_dir=split_ts, 
        partition="valid", 
        random_seed=random_seed, 
        pin_memory=True, 
        normalize=True,
        shuffle=True)

    test_dataset = TimeWindowsDataset(
        data_dir=split_ts, 
        partition="test", 
        random_seed=random_seed, 
        pin_memory=True, 
        normalize=True,
        shuffle=True)

    print("train dataset: {}".format(train_dataset))
    print("valid dataset: {}".format(valid_dataset))
    print("test dataset: {}".format(test_dataset))

    batch_size = 10

    torch.manual_seed(random_seed)
    train_generator = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_generator = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_generator = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    train_features, train_labels = next(iter(train_generator))
    print(f"Feature batch shape: {train_features.size()}; mean {torch.mean(train_features)}")
    print(f"Labels batch shape: {train_labels.size()}; mean {torch.mean(torch.Tensor.float(train_labels))}")

    gcn = GCN(graph.edge_index, 
        graph.edge_attr, 
        n_roi=X.shape[1],
        batch_size=batch_size,
        n_timepoints=1, 
        n_classes=len(main_conditions))
    print(gcn)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(gcn.parameters(), lr=1e-4, weight_decay=5e-4)

    epochs = 100
    for t in range(epochs):
        print(f"Epoch {t+1}/{epochs}\n-------------------------------")
        train_loop(train_generator, gcn, loss_fn, optimizer)
        loss, correct = valid_test_loop(valid_generator, gcn, loss_fn)
        print(f"Valid metrics:\n\t avg_loss: {loss:>8f};\t avg_accuracy: {(100*correct):>0.1f}%")

    loss, correct = valid_test_loop(test_generator, gcn, loss_fn)
    print(f"Test metrics:\n\t avg_loss: {loss:>f};\t avg_accuracy: {(100*correct):>0.1f}%")
