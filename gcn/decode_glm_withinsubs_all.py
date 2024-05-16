from glob import glob
import os
import numpy as np
from nilearn import image
from nilearn.interfaces.fmriprep import load_confounds
from tqdm import tqdm
import pandas as pd
from joblib import Parallel, delayed, dump, load
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
from torch.utils.data import Dataset
from gcn_model import GCN
import nilearn.connectome
from graph_construction import make_group_graph
import sys
from sklearn.model_selection import (
    ShuffleSplit,
    StratifiedShuffleSplit,
    train_test_split,
)
import time
from nilearn import datasets
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import seaborn as sns
import utils


warnings.filterwarnings("ignore")


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    losses = []
    accuracies = []
    for batch, (X, y) in enumerate(dataloader):
        X = X.float()
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
        losses.append(loss)
        accuracies.append(correct)
        print(
            f"#{batch:>5};\ttrain_loss: {loss:>0.3f};\ttrain_accuracy:{(100*correct):>5.1f}%\t\t[{current:>5d}/{size:>5d}]"
        )

    return np.mean(losses), np.mean(accuracies)


def valid_test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.float()
            pred = model.forward(X)
            loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    loss /= size
    correct /= size

    return loss, correct, pred, y


def calculate_connectome(X, subject, save_to):

    os.makedirs(save_to, exist_ok=True)
    connectome_file = f"{subject}_connectome.pkl"

    if os.path.exists(connectome_file):
        connectome = load(connectome_file)
    else:
        sub_mask = np.where(data["subjects"] == subject)[0]
        X = data["responses"][sub_mask]
        # Estimating connectomes and save for pytorch to load
        corr_measure = nilearn.connectome.ConnectivityMeasure(
            kind="correlation"
        )
        connectome = corr_measure.fit_transform([X])[0]
        dump(connectome, os.path.join(save_to, connectome_file))

    return connectome, subject


def create_buffer_dir(X, Y, data_dir, subject):
    buffer_dir = os.path.join(data_dir, "split_glm", subject)
    os.makedirs(buffer_dir, exist_ok=True)
    # clear buffer_dir
    for file in os.listdir(buffer_dir):
        os.remove(os.path.join(buffer_dir, file))

    main_conditions = np.unique(Y)

    encoded_labels = {name: i for i, name in enumerate(main_conditions)}
    labels_df = []

    for trial in range(X.shape[0]):
        label = Y[trial]
        filename = f"{trial}_{label}.npy"
        np.save(
            os.path.join(buffer_dir, filename),
            np.expand_dims(X[trial], axis=1),
        )
        labels_df.append(
            {"filename": filename, "label": encoded_labels[label]}
        )

    # labels dataframe
    labels_df = pd.DataFrame(labels_df)
    labels_df.to_csv(os.path.join(buffer_dir, "labels.csv"), index=False)

    return buffer_dir


def _create_train_test_split(X, Y, groups=None, random_state=0):
    if dataset == "aomic_faces":
        cv = StratifiedShuffleSplit(
            test_size=0.20, random_state=random_state, n_splits=20
        )
    else:
        cv = StratifiedShuffleSplit(
            test_size=0.10, random_state=random_state, n_splits=20
        )
    for train, test in cv.split(X, Y, groups=groups):
        N = train.shape[0]
        n_classes = np.unique(Y[train]).shape[0]
        train_sizes = np.geomspace(
            n_classes * 2, N - n_classes * 2, num=10, endpoint=True
        )
        left_out_N = N - train_sizes
        left_out_percs = left_out_N / N
        for left_out in left_out_percs:
            if left_out == 0:
                train_ = train.copy()
            else:
                indices = np.arange(X.shape[0])
                train_, _ = train_test_split(
                    indices[train],
                    test_size=left_out,
                    random_state=0,
                    stratify=Y[train],
                )
            yield train_, test, left_out


def decode(
    data, connectomes, subject, results_dir, classifier="GNN", random_state=0
):

    # select data for current subject
    sub_mask = np.where(data["subjects"] == subject)[0]
    X = data["responses"][sub_mask]
    Y = data["conditions"][sub_mask]
    groups = data["runs"][sub_mask]

    # get main conditions from np array Y
    main_conditions = np.unique(Y)

    # select graphs for all except the current subject
    connectomes_ = connectomes.copy()
    del connectomes_[subject]
    connectomes_ = list(connectomes_.values())
    connectomes_ = np.array(connectomes_)
    connectomes_ = np.mean(connectomes_, axis=0)
    print("Connectome shape: ", connectomes_.shape)

    # make a graph for the mean of connectomes
    graph = make_group_graph(
        [connectomes_], self_loops=False, k=8, symmetric=True
    )
    graph.edge_attr = torch.tensor(graph.edge_attr, dtype=torch.float32)
    graph.edge_index = torch.tensor(graph.edge_index, dtype=torch.int64)

    # results
    results = []
    # training metrics
    training_metrics = []

    # split data
    for train, test, left_out in _create_train_test_split(
        X, Y, groups=groups, random_state=random_state
    ):

        buffer_dir = create_buffer_dir(X, Y, data_dir, subject)

        train_dataset = TimeWindowsDataset(
            data_dir=buffer_dir,
            partition="train",
            random_seed=random_state,
            train_index=train,
            test_index=test,
            pin_memory=True,
            shuffle=True,
        )

        valid_dataset = TimeWindowsDataset(
            data_dir=buffer_dir,
            partition="valid",
            random_seed=random_state,
            train_index=train,
            test_index=test,
            pin_memory=True,
            shuffle=True,
        )

        test_dataset = TimeWindowsDataset(
            data_dir=buffer_dir,
            partition="test",
            random_seed=random_state,
            train_index=train,
            test_index=test,
            pin_memory=True,
            shuffle=True,
        )

        print("train dataset: {}".format(train_dataset))
        print("valid dataset: {}".format(valid_dataset))
        print("test dataset: {}".format(test_dataset))

        batch_size = len(main_conditions)

        torch.manual_seed(random_state)
        train_generator = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        valid_generator = DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=True
        )
        test_generator = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True
        )
        train_features, train_labels = next(iter(train_generator))
        print(
            f"Feature batch shape: {train_features.size()}; mean {torch.mean(train_features)}"
        )
        print(
            f"Labels batch shape: {train_labels.size()}; mean {torch.mean(torch.Tensor.float(train_labels))}"
        )

        gcn = GCN(
            graph.edge_index,
            graph.edge_attr,
            n_roi=X.shape[1],
            batch_size=batch_size,
            n_timepoints=1,
            n_classes=len(main_conditions),
        )
        print(gcn)

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            gcn.parameters(), lr=1e-4, weight_decay=5e-4
        )

        epochs = 50
        train_losses = []
        train_accuracies = []
        valid_losses = []
        valid_accuracies = []

        for t in range(epochs):
            print(f"Epoch {t+1}/{epochs}\n-------------------------------")
            train_loss, train_accuracy = train_loop(
                train_generator, gcn, loss_fn, optimizer
            )
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            valid_loss, valid_accuracy, _, _ = valid_test_loop(
                valid_generator, gcn, loss_fn
            )
            valid_losses.append(valid_loss)
            valid_accuracies.append(valid_accuracy)
            print(
                f"Valid metrics:\n\t avg_loss: {valid_loss:>8f};\t avg_accuracy: {(100*valid_accuracy):>0.1f}%"
            )

        test_loss, test_accuracy, pred, true = valid_test_loop(
            test_generator, gcn, loss_fn
        )
        balanced_accuracy = balanced_accuracy_score(true, pred.argmax(1))
        print(
            f"Test metrics:\n\t avg_loss: {test_loss:>f};\t avg_accuracy: {(100*test_accuracy):>0.1f}%"
        )
        results.append(
            {
                "subject": subject,
                "accuracy": test_accuracy,
                "balanced_accuracy": balanced_accuracy,
                "predicted": pred.argmax(1),
                "true": true,
                "train_size": len(train),
                "left_out": left_out,
                "classifier": classifier,
                "setting": "conventional",
                "dummy_accuracy": 1 / len(main_conditions),
                "dummy_balanced_accuracy": 1 / len(main_conditions),
            }
        )

        training_metrics.append(
            {
                "train_loss": train_losses,
                "train_accuracy": train_accuracies,
                "valid_loss": valid_losses,
                "valid_accuracy": valid_accuracies,
                "pred_prob": pred,
            }
        )

    results = pd.DataFrame(results)
    results.to_pickle(
        os.path.join(results_dir, f"results_clf_{classifier}_{subject}.pkl")
    )

    training_metrics = pd.DataFrame(training_metrics)
    training_metrics.to_pickle(
        os.path.join(results_dir, f"training_metrics_{subject}.pkl")
    )
    return results


if __name__ == "__main__":

    N_JOBS = 10

    DATA_ROOT = "/storage/store2/work/haggarwa/retreat_2023/data/"
    OUT_ROOT = "/storage/store2/work/haggarwa/retreat_2023/gcn/results/"

    # datasets and classifiers to use
    datas = [
        # "bold5000_fold2",
        # "bold5000_fold3",
        # "bold5000_fold4",
        # "neuromod",
        # "aomic_gstroop",
        # "forrest",
        # "rsvp",
        "aomic_anticipation",
        # "bold5000_fold1",
        # "aomic_faces",
        # "hcp_gambling",
        # "bold",
        # "nsd",
        # "ibc_aomic_gstroop",
        # "ibc_hcp_gambling",
    ]

    for dataset in datas:

        # input data root path
        data_dir = os.path.join(DATA_ROOT, dataset)
        data_resolution = "3mm"  # or 1_5mm
        nifti_dir = os.path.join(data_dir, data_resolution)

        # get difumo atlas
        atlas = datasets.fetch_atlas_difumo(
            dimension=1024,
            resolution_mm=3,
            data_dir=DATA_ROOT,
            legacy_format=False,
        )
        atlas["name"] = "difumo"

        # output results path
        start_time = time.strftime("%Y%m%d-%H%M%S")
        results_dir = f"{dataset}_{atlas.name}_results_{start_time}"
        results_dir = os.path.join(OUT_ROOT, results_dir)
        os.makedirs(results_dir, exist_ok=True)

        # get file names
        imgs = glob(os.path.join(nifti_dir, "*.nii.gz"))
        subjects = [os.path.basename(img).split(".")[0] for img in imgs]

        print(f"\nParcellating {dataset}...")
        data = Parallel(n_jobs=N_JOBS, verbose=11, backend="multiprocessing")(
            delayed(utils.parcellate)(
                imgs[i],
                subject,
                atlas,
                data_dir=data_dir,
                nifti_dir=nifti_dir,
            )
            for i, subject in enumerate(subjects)
        )

        print(f"\nConcatenating all data for {dataset}...")
        data_ = dict(responses=[], conditions=[], runs=[], subjects=[])
        for entry in data:
            for key in ["responses", "conditions", "runs", "subjects"]:
                data_[key].extend(entry[key])
        for key in ["responses", "conditions", "runs", "subjects"]:
            data_[key] = np.array(data_[key])
        data = data_.copy()
        del data_

        print(f"Calculating connectomes for {dataset}...")
        connectomes = Parallel(
            n_jobs=N_JOBS * 3, verbose=11, backend="multiprocessing"
        )(
            delayed(calculate_connectome)(
                data, subject, os.path.join(data_dir, "connectomes")
            )
            for subject in subjects
        )

        connectomes = {
            subject: connectome for connectome, subject in connectomes
        }

        print(f"Decoding {dataset}...")
        all_results = Parallel(
            n_jobs=N_JOBS * 3, verbose=11, backend="multiprocessing"
        )(
            delayed(decode)(
                data,
                connectomes,
                subject,
                results_dir,
                classifier="GNN",
                random_state=0,
            )
            for subject in subjects
        )

        # all_results = []
        # for subject in subjects:
        #     print(f"Decoding {subject}...")
        #     sub_res = decode(
        #         data,
        #         connectomes,
        #         subject,
        #         results_dir,
        #         classifier="GNN",
        #         random_state=0,
        #     )
        #     all_results.append(sub_res)

        print(f"Results saved in {results_dir}")

        df = pd.concat(all_results)
        sns.pointplot(
            data=df,
            x="train_size",
            y="balanced_accuracy",
            hue="classifier",
        )
        plt.axhline(
            y=df["dummy_balanced_accuracy"].mean(), color="k", linestyle="--"
        )
        plt.ylabel("Balanced Accuracy")
        plt.xlabel("Training size")
        plt.savefig(os.path.join(results_dir, "balanced_accuracy.png"))
        plt.close()
