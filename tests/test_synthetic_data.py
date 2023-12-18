from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

WORK_DIR = Path(os.environ["HOME"]) / "BRICS-TB/tb-brics-tools/screening"
DATA_DIR = WORK_DIR / "explore/BackDoor/data"


def load_metadata(files: dict[str, Path]) -> pd.DataFrame:
    label_mapper = {"tb": True, "notb": False}
    metadata_list = []
    for label in files:
        filepath = files[label]
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
        metadata_list.append(data)

    return pd.concat(metadata_list)


class TestShenzhenWGAN:
    @pytest.fixture
    def china(self):
        files = {
            "tb": DATA_DIR
            / "Shenzhen/china/fake_images/user.joao.pinto.task.Shenzhen_china.wgan.v2_tb.r1.samples.csv",
            "notb": DATA_DIR
            / "Shenzhen/china/fake_images/user.joao.pinto.task.Shenzhen_china.wgan.v2_notb.r1.samples.csv",
        }
        return load_metadata(files)

    def test_folds(self, china: pd.DataFrame) -> None:
        assert set(china["fold"].unique()) == set(range(0, 10))

    def test_inner_folds(self, china: pd.DataFrame) -> None:
        assert set(china["inner_fold"].unique()) == set(range(0, 9))

    def test_sets(self, china: pd.DataFrame) -> None:
        assert set(china["set"].unique()) == set(["test"])

    def get_n_images(
        self, china: pd.DataFrame, fold: int, inner_fold: int
    ) -> pd.DataFrame:
        df = china[(china["fold"] == fold) & (china["inner_fold"] == inner_fold)]
        return df.shape[0]

    def test_n_images(self, china: pd.DataFrame) -> None:
        for fold in range(10):
            for inner_fold in range(9):
                n_images = self.get_n_images(china, fold, inner_fold)
                # assert (n_images >= 6400) & (n_images <= 6418)
                assert n_images == 6400

    def get_n_labels(
        self, china: pd.DataFrame, fold: int, inner_fold: int
    ) -> dict[str, int]:
        labels = china[(china["fold"] == fold) & (china["inner_fold"] == inner_fold)][
            "label"
        ].astype(str)
        return labels.value_counts().to_dict()

    def test_n_labels(self, china: pd.DataFrame) -> None:
        for fold in range(10):
            for inner_fold in range(9):
                labels = self.get_n_labels(china, fold, inner_fold)
                # assert labels["True"] == labels["False"]
                assert labels["True"] == 3200
                assert labels["False"] == 3200


class TestShenzhenPix2Pix:
    @pytest.fixture
    def china(self):
        files = {
            "tb": DATA_DIR
            / "Shenzhen/china/fake_images/user.otto.tavares.task.Shenzhen_china.pix2pix.v1_tb.r1.samples.csv",
            "notb": DATA_DIR
            / "Shenzhen/china/fake_images/user.otto.tavares.task.Shenzhen_china.pix2pix.v1_notb.r1.samples.csv",
        }
        return load_metadata(files)

    def test_folds(self, china: pd.DataFrame) -> None:
        assert set(china["fold"].unique()) == set(range(0, 10))

    def test_inner_folds(self, china: pd.DataFrame) -> None:
        assert set(china["inner_fold"].unique()) == set(range(0, 9))

    def test_sets(self, china: pd.DataFrame) -> None:
        assert set(china["set"].unique()) == set(["train", "val", "test"])

    def test_n_images(self, china: pd.DataFrame) -> None:
        n_images = (china["path"].apply(lambda x: x.split("/")[-1])).nunique()
        assert n_images == 566

    def test_n_labels(self, china: pd.DataFrame) -> None:
        china["filename"] = china["path"].apply(lambda x: x.split("/")[-1])
        labels = china[["filename", "label"]].drop_duplicates()["label"]
        labels = labels.astype(str).value_counts().to_dict()
        assert labels["True"] == 287
        assert labels["False"] == 279


class TestManausWGAN:
    @pytest.fixture
    def manaus(self):
        files = {
            "tb": DATA_DIR
            / "Manaus/manaus/fake_images/user.joao.pinto.task.Manaus.manaus.wgan_v2_tb.r2.sample.csv",
            "notb": DATA_DIR
            / "Manaus/manaus/fake_images/user.joao.pinto.task.Manaus.manaus.wgan_v2_notb.r2.sample.csv",
        }
        return load_metadata(files)

    def test_folds(self, manaus: pd.DataFrame) -> None:
        assert set(manaus["fold"].unique()) == set(range(0, 10))

    def test_inner_folds(self, manaus: pd.DataFrame) -> None:
        assert set(manaus["inner_fold"].unique()) == set(range(0, 9))

    def test_sets(self, manaus: pd.DataFrame) -> None:
        assert set(manaus["set"].unique()) == set(["test"])

    def get_n_images(
        self, manaus: pd.DataFrame, fold: int, inner_fold: int
    ) -> pd.DataFrame:
        df = manaus[(manaus["fold"] == fold) & (manaus["inner_fold"] == inner_fold)]
        return df.shape[0]

    def test_n_images(self, manaus: pd.DataFrame) -> None:
        for fold in range(10):
            for inner_fold in range(9):
                n_images = self.get_n_images(manaus, fold, inner_fold)
                assert n_images == 6400

    def get_n_labels(
        self, manaus: pd.DataFrame, fold: int, inner_fold: int
    ) -> dict[str, int]:
        labels = manaus[
            (manaus["fold"] == fold) & (manaus["inner_fold"] == inner_fold)
        ]["label"].astype(str)
        return labels.value_counts().to_dict()

    def test_n_labels(self, manaus: pd.DataFrame) -> None:
        for fold in range(10):
            for inner_fold in range(9):
                labels = self.get_n_labels(manaus, fold, inner_fold)
                assert labels["True"] == labels["False"]


class TestManausPix2Pix:
    @pytest.fixture
    def manaus(self):
        files = {
            "tb": DATA_DIR
            / "Manaus/manaus/fake_images/user.otto.tavares.Manaus.manaus.pix2pix_v1.tb.r3.samples.csv",
            "notb": DATA_DIR
            / "Manaus/manaus/fake_images/user.otto.tavares.Manaus.manaus.pix2pix_v1.notb.r3.samples.csv",
        }
        return load_metadata(files)

    def test_folds(self, manaus: pd.DataFrame) -> None:
        assert set(manaus["fold"].unique()) == set(range(0, 10))

    def test_inner_folds(self, manaus: pd.DataFrame) -> None:
        assert set(manaus["inner_fold"].unique()) == set(range(0, 9))

    def test_sets(self, manaus: pd.DataFrame) -> None:
        assert set(manaus["set"].unique()) == set(["train", "val", "test"])

    def test_n_images(self, manaus: pd.DataFrame) -> None:
        n_images = (manaus["path"].apply(lambda x: x.split("/")[-1])).nunique()
        assert n_images == 120

    def test_n_labels(self, manaus: pd.DataFrame) -> None:
        manaus["filename"] = manaus["path"].apply(lambda x: x.split("/")[-1])
        labels = manaus[["filename", "label"]].drop_duplicates()["label"]
        labels = labels.astype(str).value_counts().to_dict()
        assert labels["True"] == 23
        assert labels["False"] == 97


class TestManausCycle:
    @pytest.fixture
    def manaus(self):
        files = {
            "tb": DATA_DIR
            / "Manaus/manaus/fake_images/user.otto.tavares.Manaus.manaus.cycle_v1_tb.r2.Manaus_to_SantaCasa.samples.csv",
            "notb": DATA_DIR
            / "Manaus/manaus/fake_images/user.otto.tavares.Manaus.manaus.cycle_v1_notb.r2.SantaCasa_to_Manaus.samples.csv",
        }
        return load_metadata(files)

    def test_folds(self, manaus: pd.DataFrame) -> None:
        assert set(manaus["fold"].unique()) == set(range(0, 10))

    def test_inner_folds(self, manaus: pd.DataFrame) -> None:
        assert set(manaus["inner_fold"].unique()) == set(range(0, 9))

    def test_sets(self, manaus: pd.DataFrame) -> None:
        assert set(manaus["set"].unique()) == set(["train", "val", "test"])

    def test_n_images(self, manaus: pd.DataFrame) -> None:
        n_images = (manaus["path"].apply(lambda x: x.split("/")[-1])).nunique()
        assert n_images == 206

    def test_n_labels(self, manaus: pd.DataFrame) -> None:
        manaus["filename"] = manaus["path"].apply(lambda x: x.split("/")[-1])
        labels = manaus[["filename", "label"]].drop_duplicates()["label"]
        labels = labels.astype(str).value_counts().to_dict()
        assert labels["True"] == 39
        assert labels["False"] == 167


class TestSantaCasaWGAN:
    @pytest.fixture
    def imageamento_anonimizado_valid(self):
        files = {
            "notb": DATA_DIR
            / "SantaCasa/imageamento_anonimizado_valid/fake_images/user.joao.pinto.task.SantaCasa.imageamento_anonimizado_valid.wgan_v2_notb.r4.samples.csv"
        }
        return load_metadata(files)

    def test_folds(self, imageamento_anonimizado_valid: pd.DataFrame) -> None:
        assert set(imageamento_anonimizado_valid["fold"].unique()) == set(range(0, 10))

    def test_inner_folds(self, imageamento_anonimizado_valid: pd.DataFrame) -> None:
        assert set(imageamento_anonimizado_valid["inner_fold"].unique()) == set(
            range(0, 9)
        )

    def test_sets(self, imageamento_anonimizado_valid: pd.DataFrame) -> None:
        assert set(imageamento_anonimizado_valid["set"].unique()) == set(["test"])

    def get_n_images(
        self, imageamento_anonimizado_valid: pd.DataFrame, fold: int, inner_fold: int
    ) -> pd.DataFrame:
        df = imageamento_anonimizado_valid[
            (imageamento_anonimizado_valid["fold"] == fold)
            & (imageamento_anonimizado_valid["inner_fold"] == inner_fold)
        ]
        return df.shape[0]

    def test_n_images(self, imageamento_anonimizado_valid: pd.DataFrame) -> None:
        for fold in range(10):
            for inner_fold in range(9):
                n_images = self.get_n_images(
                    imageamento_anonimizado_valid, fold, inner_fold
                )
                assert n_images == 3200

    def get_n_labels(
        self, imageamento_anonimizado_valid: pd.DataFrame, fold: int, inner_fold: int
    ) -> str:
        labels = imageamento_anonimizado_valid[
            (imageamento_anonimizado_valid["fold"] == fold)
            & (imageamento_anonimizado_valid["inner_fold"] == inner_fold)
        ]["label"].astype(str)
        return labels.unique()

    def test_labels(self, imageamento_anonimizado_valid: pd.DataFrame) -> None:
        for fold in range(10):
            for inner_fold in range(9):
                labels = self.get_n_labels(
                    imageamento_anonimizado_valid, fold, inner_fold
                )
                assert labels == "False"


class TestSantaCasaPix2Pix:
    @pytest.fixture
    def imageamento_anonimizado_valid(self):
        files = {
            "notb": DATA_DIR
            / "SantaCasa/imageamento_anonimizado_valid/fake_images/user.otto.tavares.task.SantaCasa_imageamento_anonimizado_valid.pix2pix_v1_notb.r2.samples.csv"
        }
        return load_metadata(files)

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
        assert n_images == 149

    def test_n_labels(self, imageamento_anonimizado_valid: pd.DataFrame) -> None:
        imageamento_anonimizado_valid["filename"] = imageamento_anonimizado_valid[
            "path"
        ].apply(lambda x: x.split("/")[-1])
        labels = imageamento_anonimizado_valid[["filename", "label"]].drop_duplicates()[
            "label"
        ]
        labels = labels.astype(str).value_counts().to_dict()
        assert labels["False"] == 149


class TestSantaCasaCycle:
    @pytest.fixture
    def imageamento_anonimizado_valid(self):
        files = {
            "tb": DATA_DIR
            / "SantaCasa/imageamento_anonimizado_valid/fake_images/user.otto.tavares.task.SantaCasa_imageamento_anonimizado_valid.cycle_v1_tb.r1.Shenzhen_to_SantaCasa.samples.csv",
            "notb": DATA_DIR
            / "SantaCasa/imageamento_anonimizado_valid/fake_images/user.otto.tavares.task.SantaCasa_imageamento_anonimizado_valid.cycle_v1_notb.r1.SantaCasa_to_Shenzhen.samples.csv",
        }
        return load_metadata(files)

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
        assert n_images == 503

    def test_n_labels(self, imageamento_anonimizado_valid: pd.DataFrame) -> None:
        imageamento_anonimizado_valid["filename"] = imageamento_anonimizado_valid[
            "path"
        ].apply(lambda x: x.split("/")[-1])
        labels = imageamento_anonimizado_valid[["filename", "label"]].drop_duplicates()[
            "label"
        ]
        labels = labels.astype(str).value_counts().to_dict()
        assert labels["True"] == 336
        assert labels["False"] == 167
