import os.path
import random
import subprocess

import mne
import numpy as np
import pandas as pd
import torch
import shlex
import sys

from sklearn.utils.extmath import _approximate_mode


def load_metadata_database(data_path):
    csv_path = os.path.join(data_path, 'used_label_database.csv')
    database = pd.read_csv(
                csv_path,
                delimiter="|",
                index_col=0,
                encoding="utf8",
                low_memory=False,
    )
    return database


def read_folds_override(data_path, folds_path=None):
    """Returns all possible folds names and folds database"""
    if folds_path is None:
        folds_path = os.path.join(data_path, 'folds.csv')
    elif os.path.exists(folds_path):
        folds_path = os.path.abspath(folds_path)
    else:
        raise FileNotFoundError(folds_path)
    folds = pd.read_csv(folds_path, index_col=0, header=None, names=["fold", ])
    folds_list = list(sorted(folds["fold"].unique()))
    return folds_list, folds


def get_eids_for_folds(folds_db, folds=1):
    if not isinstance(folds, list):
        folds = [folds, ]

    eids = []
    for fold in folds:
        eids_in_fold = [str(i) for i in folds_db[folds_db['fold'] == fold].index.values]
        eids.extend(eids_in_fold)
    return eids


def load_label_override(label_override, data_path):
    """Loads a database CSV column as a pseudo dict
    label_override - string ie 'column_name:path_to.csv' or just column name,
    then path will be taken from data_path + 'used_label_database'

    data_path - string for preprocessed data folder, used if you only want to change the column, but want to use default csv
    """
    if len(label_override) == 0:
        return None

    label_override_split = label_override.split(':')
    if len(label_override_split) == 1:
        column_name = label_override
        csv_path = os.path.join(data_path, 'used_label_database.csv')
    elif len(label_override_split) == 2:
        column_name, csv_path = label_override_split
    else:
        raise Exception("Incorrect use of label_override, fix this:", label_override)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Provided CSV {csv_path} does not exist!")
    # print(f"Using label override! Loading column {column_name} from csv {csv_path}")
    database = pd.read_csv(
        csv_path,
        delimiter="|",
        index_col='examination_id',
        encoding="utf8",
    )

    return database[column_name]  # if it's one column it becomes a series, directly identifieable by eid

def label_override(eids, label_override_dict):
    '''
    eids - list of strings, examination_ids
    label_override_dict - pseudo dict loaded using load_label_override

    returns - numpy array of float labels, potentially fuzzy
    '''
    labels = []
    for eid in eids:
        label = label_override_dict[eid]
        # labels can be floats
        # or strings
        if isinstance(label, str):
            if label == 'normal':
                label = 0.0
            else:
                label = 1.0
        labels.append(label)
    return np.array(labels)

def overwrite_prompt(folder):
    folder = os.path.abspath(folder)
    answer = ''
    while answer not in ['y', 'n']:
        print(f"WARNING! {folder} exists.\n"
              f"If you proceed the contents will be DESTROYED FOREVER\n"
              f"Do you want to DELETE and OVERWRITE {folder}?\n"
              f"y/n")
        try:
            answer = input()
        except Exception:
            continue
        if answer == 'y':
            break
        if answer == 'n':
            return False

    folder_name = os.path.basename(folder)
    print(f"Are you sure you want to delete {folder}?\n"
          f"If so type it's name:\n{folder_name}")
    answer = ''
    count = 3
    while answer != folder_name and count > 0:
        try:
            answer = input().strip()
        except Exception:
            continue
        if answer != folder_name:
            count -= 1
        if answer == folder_name:
            return True
    return False


def seed_everything(seed: int) -> int:
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

    return seed


def exists(val):
    return val is not None

def chunk(to_chunk: list, step: int):
    chunked = []
    for i in range(0, len(to_chunk), step):
        chunked.append(to_chunk[i: i + step])

    return chunked


def stratified_sample(n_to_sample, data, frame_timings, classes_all):
    # takes data, where 0th axis is samples, list of classes, and how many samples you want to get
    # returns stratified subsample of data and classes
    # based on the code of sklearn.model_selection.StratifiedShuffleSplit
    classes, y_indices = np.unique(classes_all, return_inverse=True)
    n_classes = classes.shape[0]

    class_counts = np.bincount(y_indices)
    class_indices = np.split(
        np.argsort(y_indices, kind="mergesort"), np.cumsum(class_counts)[:-1]
    )
    rng = np.random.RandomState()
    n_i = _approximate_mode(class_counts, n_to_sample, rng=rng)

    selected_indices = []

    for i in range(n_classes):
        permutation = rng.permutation(class_counts[i])
        perm_indices_class_i = class_indices[i].take(permutation, mode="clip")

        selected_indices.extend(perm_indices_class_i[: n_i[i]])

    selected_indices = rng.permutation(selected_indices)
    data_filtered = data[selected_indices]
    frame_timings = np.array(frame_timings)[selected_indices]
    classes_filtered = np.array([classes_all[i] for i in selected_indices])
    IDX_sort = np.argsort([i[0] for i in frame_timings])
    return data_filtered[IDX_sort], classes_filtered[IDX_sort].tolist(), frame_timings[IDX_sort].tolist()
