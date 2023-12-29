import os
import sys
import hashlib
import numpy as np
import pandas as pd
import pydicom as dicom

from pathlib import Path
from itertools import product
from joblib import Memory
from PIL import Image
from glob import glob
from sklearn.model_selection import StratifiedKFold

MEMORY     = Memory("/tmp/LTBI")
DATA_DIR   = Path(os.environ["DATA_DIR"])


@MEMORY.cache
def get_filename(
    data_path: str,
    dataset_name: str,
) -> pd.DataFrame:
    """Search for available images.

    Args:
        data_path (str): Data directory containing the images.
        dataset_name (str): Name of the dataset.
            - 'shenzhen'
            - 'montgomery'
            - 'india'

    Returns:
        pd.DataFrame: Dataframe columns: 'path', 'labels' and 'hash'.
    """
    if dataset_name == "shenzhen":
        H0 = glob(os.path.join(data_path, "CHNCXR_*_0.png"))
        H1 = glob(os.path.join(data_path, "CHNCXR_*_1.png"))

        n_H0 = len(H0)
        n_H1 = len(H1)
        label = list(np.zeros(n_H0)) + list(np.ones(n_H1))

        d = {"path": H0 + H1, "label": label}

    elif dataset_name == "montgomery":
        H0 = glob(os.path.join(data_path, "MCUCXR_*_0.png"))
        H1 = glob(os.path.join(data_path, "MCUCXR_*_1.png"))

        n_H0 = len(H0)
        n_H1 = len(H1)
        label = list(np.zeros(n_H0)) + list(np.ones(n_H1))

        d = {"path": H0 + H1, "label": label}

    elif dataset_name == "india":
        H0 = glob(os.path.join(data_path, "n*.jpg"))
        H1 = glob(os.path.join(data_path, "p*.jpg"))

        n_H0 = len(H0)
        n_H1 = len(H1)
        label = list(np.zeros(n_H0)) + list(np.ones(n_H1))

        d = {"path": H0 + H1, "label": label}

    elif dataset_name == "montgomery":
        H0 = glob(os.path.join(data_path, "MCUCXR_*_0.png"))
        H1 = glob(os.path.join(data_path, "MCUCXR_*_1.png"))
    elif dataset_name == "india":
        H0_A = glob(os.path.join(data_path, "nx*.jpg"))
        H0_B = glob(os.path.join(data_path, "n*.dcm"))
        H1_A = glob(os.path.join(data_path, "px*.jpg"))
        H1_B = glob(os.path.join(data_path, "p*.dcm"))
        H0 = H0_A + H0_B
        H1 = H1_A + H1_B
    else:
        print("Not a valid dataset name.")
        sys.exit()

    n_H0 = len(H0)
    n_H1 = len(H1)

    label = list(np.zeros(n_H0)) + list(np.ones(n_H1))

    d = {"path": H0 + H1, "label": label}

    # get image hashes
    hash_aux = []
    for i in d["path"]:
        if i.endswith(".dcm"):
            image_meta = dicom.dcmread(i, force=True)
            image = image_meta.pixel_array
            hash_aux.append(hashlib.md5(image.tobytes()))
        else:
            hash_aux.append(hashlib.md5(Image.open(i).tobytes()))

    hashes = [hash.hexdigest() for hash in hash_aux]
    d["hash"] = hashes

    return pd.DataFrame(d)


def cross_validation(
    data_info: pd.DataFrame,
    n_splits: int,
    seed: int ) -> dict:

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    data_trn = []; data_tst = []
    for trn_idx, tst_idx in cv.split(data_info["path"], data_info["label"]):
        data_trn.append(data_info.iloc[trn_idx])
        data_tst.append(data_info.iloc[tst_idx])

    cv_runs = {"train": data_trn, "test": data_tst}

    return cv_runs




#
# prepare real data
#
def prepare_real( dataset : str, tag : str, metadata: dict ) -> pd.DataFrame:

     path = DATA_DIR / f"{dataset}/{tag}/raw"
     filepath = path / metadata["csv"]
     if not filepath.is_file():
         raise FileNotFoundError(f"File {filepath} not found.")

     data = pd.read_csv(filepath).rename(
         columns={"target": "label", "image_path": "path"}
     )

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
        data["label"] = label_mapper[label]
        data["type"] = "fake"
        data["name"] = dataset
        data["source"] = "pix2pix"
        metadata_list.append(data)
    return pd.concat(metadata_list)


#
# prepare wgan data
#
def prepare_wgan(dataset : str, tag : str, metadata: dict) -> pd.DataFrame:

    path = DATA_DIR / f"{dataset}/{tag}/fake_images"
    label_mapper = {"tb": True, "notb": False}
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
        data["label"] = label_mapper[label]
        data["type"] = "fake"
        data["name"] = dataset
        data["source"] = "wgan"
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
