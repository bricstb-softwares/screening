from __future__ import annotations

import os
from itertools import product
from pathlib import Path

import pandas as pd
import pytest

WORKDIR = Path(os.environ["HOME"]) / "BRICS-TB/tb-brics-tools/screening"
DATADIR = WORKDIR / "explore/BackDoor/data"


def load_metadata(file: Path, filename: str) -> pd.DataFrame:
    filepath = file / filename
    if not filepath.is_file():
        raise FileNotFoundError(f"File {filepath} not found.")
    data = pd.read_csv(filepath, usecols=["image_path", "target"])
    data.rename(columns={"target": "label", "image_path": "path"}, inplace=True)
    data["label"] = data["label"].astype(bool)
    data["type"] = "real"

    filepath = file / "splits.pkl"
    if not filepath.is_file():
        raise FileNotFoundError(f"The file {filepath} does not exist.")
    splits = pd.read_pickle(filepath)

    folds = list(range(len(splits)))
    inner_folds = list(range(len(splits[0])))

    real_list = []
    for i, j in product(folds, inner_folds):
        trn_idx = splits[i][j][0]
        val_idx = splits[i][j][1]
        tst_idx = splits[i][j][2]

        train = data.loc[trn_idx, ["path", "label", "type"]]
        train["set"] = "train"
        train["fold"] = i
        train["inner_fold"] = j
        real_list.append(train)

        valid = data.loc[val_idx, ["path", "label", "type"]]
        valid["set"] = "val"
        valid["fold"] = i
        valid["inner_fold"] = j
        real_list.append(valid)

        test = data.loc[tst_idx, ["path", "label", "type"]]
        test["set"] = "test"
        test["fold"] = i
        test["inner_fold"] = j
        real_list.append(test)

    return pd.concat(real_list)


class TestShenzhen:
    @pytest.fixture
    def china(self) -> pd.DataFrame:
        file = DATADIR / "Shenzhen/china/raw"
        filename = "Shenzhen_china_table_from_raw.csv"
        return load_metadata(file, filename)

    def test_folds(self, china: pd.DataFrame) -> None:
        assert set(china["fold"].unique()) == set(range(0, 10))

    def test_inner_folds(self, china: pd.DataFrame) -> None:
        assert set(china["inner_fold"].unique()) == set(range(0, 9))

    def test_sets(self, china: pd.DataFrame) -> None:
        assert set(china["set"].unique()) == set(["train", "val", "test"])

    def test_n_images(self, china: pd.DataFrame) -> None:
        n_images = (china["path"].apply(lambda x: x.split("/")[-1])).nunique()
        assert n_images == 662

    def test_n_labels(self, china: pd.DataFrame) -> None:
        china["filename"] = china["path"].apply(lambda x: x.split("/")[-1])
        labels = china[["filename", "label"]].drop_duplicates()["label"]
        labels = labels.astype(str).value_counts().to_dict()
        assert labels["True"] == 336
        assert labels["False"] == 326


class TestManaus:
    @pytest.fixture
    def manaus(self) -> pd.DataFrame:
        file = DATADIR / "Manaus/c_manaus/raw"
        filename = "Manaus_c_manaus_table_from_raw.csv"
        return load_metadata(file, filename)

    def test_folds(self, manaus: pd.DataFrame) -> None:
        assert set(manaus["fold"].unique()) == set(range(0, 10))

    def test_inner_folds(self, manaus: pd.DataFrame) -> None:
        assert set(manaus["inner_fold"].unique()) == set(range(0, 9))

    def test_sets(self, manaus: pd.DataFrame) -> None:
        assert set(manaus["set"].unique()) == set(["train", "val", "test"])

    def test_n_images(self, manaus: pd.DataFrame) -> None:
        n_images = (manaus["path"].apply(lambda x: x.split("/")[-1])).nunique()
        assert n_images == 700

    def test_n_labels(self, manaus: pd.DataFrame) -> None:
        manaus["filename"] = manaus["path"].apply(lambda x: x.split("/")[-1])
        labels = manaus[["filename", "label"]].drop_duplicates()["label"]
        labels = labels.astype(str).value_counts().to_dict()
        assert labels["True"] == 167
        assert labels["False"] == 533


class TestSantaCasa:
    @pytest.fixture
    def imageamento_anonimizado_valid(self) -> pd.DataFrame:
        file = DATADIR / "SantaCasa/imageamento_anonimizado_valid/raw"
        filename = "SantaCasa_imageamento_anonimizado_valid_table_from_raw.csv"
        return load_metadata(file, filename)

    def test_folds(self, imageamento_anonimizado_valid: pd.DataFrame) -> None:
        assert set(imageamento_anonimizado_valid["fold"].unique()) == set(range(0, 10))

    def test_inner_folds(self, imageamento_anonimizado_valid: pd.DataFrame) -> None:
        assert set(imageamento_anonimizado_valid["inner_fold"].unique()) == set(
            range(0, 9)
        )

    def test_sets(self, imageamento_anonimizado_valid: pd.DataFrame) -> None:
        assert set(imageamento_anonimizado_valid["set"].unique()) == set(
            ["train", "val", "test"]
        )

    def test_n_images(self, imageamento_anonimizado_valid: pd.DataFrame) -> None:
        n_images = (
            imageamento_anonimizado_valid["path"].apply(lambda x: x.split("/")[-1])
        ).nunique()
        assert n_images == 167

    def test_n_labels(self, imageamento_anonimizado_valid: pd.DataFrame) -> None:
        imageamento_anonimizado_valid["filename"] = imageamento_anonimizado_valid[
            "path"
        ].apply(lambda x: x.split("/")[-1])
        labels = imageamento_anonimizado_valid[["filename", "label"]].drop_duplicates()[
            "label"
        ]
        labels = labels.astype(str).value_counts().to_dict()
        assert labels["False"] == 167
