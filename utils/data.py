import os
import sys
import hashlib
import numpy as np
import pandas as pd
import pydicom as dicom

from joblib import Memory
from PIL import Image
from glob import glob
from sklearn.model_selection import StratifiedKFold

MEMORY = Memory("/tmp/LTBI")


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
    seed: int,
) -> dict:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    data_trn = []
    data_tst = []
    for trn_idx, tst_idx in cv.split(data_info["path"], data_info["label"]):
        data_trn.append(data_info.iloc[trn_idx])
        data_tst.append(data_info.iloc[tst_idx])

    cv_runs = {"train": data_trn, "test": data_tst}

    return cv_runs
