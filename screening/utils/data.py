__all__ = [
    "prepare_data",
    "split_dataframe",
]

import os, sys
import numpy as np
import pandas as pd

from pathlib import Path
from itertools import product
from sklearn.model_selection import train_test_split
from screening import DATA_DIR, TARGET_DIR
from pprint import pprint
from loguru import logger



def split_dataframe(df, fold, inner_fold, type_set):
    # type: (pd.DataFrame, int, int, str) -> pd.DataFrame
    if type_set == "train_real":
        res = df[
            (df["type"] == "real")
            & (df["set"] == "train")
            & (df["fold"] == fold)
            & (df["inner_fold"] == inner_fold)
        ]
    elif type_set == "valid_real":
        res = df[
            (df["type"] == "real")
            & (df["set"] == "val")
            & (df["fold"] == fold)
            & (df["inner_fold"] == inner_fold)
        ]
    elif type_set == "train_fake":
        res = df[
            (df["type"] == "fake")
            & (df["set"] == "train")
            & (df["fold"] == fold)
            & (df["inner_fold"] == inner_fold)
        ]
    elif type_set == "test_real":
        res = df[
            (df["type"] == "real")
            & (df["set"] == "test")
            & (df["fold"] == fold)
            & (df["inner_fold"] == inner_fold)
        ]
    else:
        raise NotImplementedError(f"Type set '{type_set}' not implemented.")

    return res[["source", "path", "label"]]



#
# prepare real data
#
def prepare_real( dataset : str, tag : str, metadata: dict ) -> pd.DataFrame:

    path = DATA_DIR / f"{dataset}/{tag}/raw"
    logger.info(metadata)
    try: # current key access
        filepath = path / metadata["csv"]
    except: # HACK: since we have files with old key access.
        filepath = path / metadata["raw"]

    if not filepath.is_file():
        raise FileNotFoundError(f"File {filepath} not found.")
    
    data = pd.read_csv(filepath).rename(
        columns={"target": "label", "image_path": "path"}
    )
    def _append_basepath(row):
       return f"{DATA_DIR}/{dataset}/{tag}/raw/{row.path}"

    data['path']   = data.apply(_append_basepath, axis='columns')
    data["name"]   = dataset
    data["type"]   = "real"
    data["source"] = "experimental"
    filepath = path / metadata["pkl"]
    if not filepath.is_file():
        raise FileNotFoundError(f"File {filepath} not found.")
    splits = pd.read_pickle(filepath)
    folds = list(range(len(splits)))
    inner_folds = list(range(len(splits[0])))
    cols = ["path", "label", "type", "name", "source"]
    metadata_list = []
    
    for i, j in product(folds, inner_folds):
        trn_idx = splits[i][j][0]
        val_idx = splits[i][j][1]
        tst_idx = splits[i][j][2]
        train = data.loc[trn_idx, cols]
        train["set"] = "train"
        train["fold"] = i
        train["inner_fold"] = j
        metadata_list.append(train)
        valid = data.loc[val_idx, cols]
        valid["set"] = "val"
        valid["fold"] = i
        valid["inner_fold"] = j
        metadata_list.append(valid)
        test = data.loc[tst_idx, cols]
        test["set"] = "test"
        test["fold"] = i
        test["inner_fold"] = j
        metadata_list.append(test)
    return pd.concat(metadata_list)


#
# prepare p2p
#
def prepare_p2p(dataset : str, tag : str, metadata: dict) -> pd.DataFrame:

    path = DATA_DIR / f"{dataset}/{tag}/fake_images"
    label_mapper = {"tb": True, "notb": False}
    metadata_list = []
    
    def _append_basepath(row):
        return f"{str(path)}/{row.path}"

    for label in metadata:
        filepath = path / metadata[label]
        if not filepath.is_file():
            raise FileNotFoundError(f"File {filepath} not found.")
        data = pd.read_csv(filepath, usecols=["image_path", "test", "sort", "type"])
        data.rename(
            columns={
                "test"      : "fold",
                "sort"      : "inner_fold",
                "type"      : "set",
                "image_path": "path",
            },
            inplace=True,
        )
        data["label"]   = label_mapper[label]
        data["type"]    = "fake"
        data["name"]    = dataset
        data["source"]  = "pix2pix"
        data['path']    = data.apply(_append_basepath, axis='columns')
    

        metadata_list.append(data)
    return pd.concat(metadata_list)


#
# prepare wgan data
#
def prepare_wgan(dataset : str, tag : str, metadata: dict) -> pd.DataFrame:

    path = DATA_DIR / f"{dataset}/{tag}/fake_images"
    label_mapper = {"tb": True, "notb": False}

    def _append_basepath(row):
        return f"{str(path)}/{row.path}"

    metadata_list = []
    for label in metadata:
        filepath = path / metadata[label]
        if not filepath.is_file():
            raise FileNotFoundError(f"File {filepath} not found.")
        data = pd.read_csv(filepath, usecols=["image_path", "test", "sort"])
        data = data.sample(n=600, random_state=42)  # sample a fraction of images
        data.rename(
            columns={
                    "test": "fold", 
                    "sort": "inner_fold", 
                    "image_path": "path"
                    },
            inplace=True,
        )
        data["label"]   = label_mapper[label]
        data["type"]    = "fake"
        data["name"]    = dataset
        data["source"]  = "wgan"
        data['path']    = data.apply(_append_basepath, axis='columns')

        metadata_list.append(data)

    data_train, data_valid = train_test_split(
        pd.concat(metadata_list), test_size=0.2, shuffle=True, random_state=512
    )
    data_train["set"] = "train"
    data_valid["set"] = "val"
    return pd.concat([data_train, data_valid])


#
# prepare cycle data
#
def prepare_cycle(dataset : str, tag : str, metadata: dict) -> pd.DataFrame:

    path = DATA_DIR / f"{dataset}/{tag}/fake_images"
    label_mapper = {"tb": True, "notb": False}

    def _append_basepath(row):
        return f"{str(path)}/{row.path}"

    metadata_list = []
    for label in metadata:
        filepath = path / metadata[label]
        if not filepath.is_file():
            raise FileNotFoundError(f"File {filepath} not found.")
        data = pd.read_csv(filepath, usecols=["image_path", "test", "sort", "type"])
        data.rename(
            columns={
                "test": "fold",
                "sort": "inner_fold",
                "type": "set",
                "image_path": "path",
            },
            inplace=True,
        )
        data["label"] = label_mapper[label]
        data["type"] = "fake"
        data["name"] = dataset
        data["source"] = "cycle"
        data['path']    = data.apply(_append_basepath, axis='columns')

        metadata_list.append(data)
    return pd.concat(metadata_list)


#
# prepare data
#
def prepare_data( source : str, dataset : str, tag : str, metadata: dict) -> pd.DataFrame:

    if source == "raw":
        data = prepare_real(dataset, tag, metadata)
    elif source == "pix2pix":
        data = prepare_p2p(dataset, tag, metadata)
    elif source == "wgan":
        data = prepare_wgan(dataset, tag, metadata)
    elif source == "cycle":
        data = prepare_cycle(dataset, tag, metadata)
    else:
        raise KeyError(f"Source '{source}' is not defined.")
    return data
