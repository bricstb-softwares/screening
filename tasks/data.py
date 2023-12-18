import os
import luigi
import pandas as pd
from sklearn.model_selection import train_test_split
from itertools import product
from luigi.format import Nop
from dotenv import load_dotenv
from pathlib import Path
from tasks.commons import Task


load_dotenv()
DATA_DIR   = Path(os.environ["DATA_DIR"])
TARGET_DIR = Path(os.environ["TARGET_DIR"])


class CrossValidation(Task):
    dataset = luigi.Parameter()
    tag     = luigi.Parameter()
    source  = luigi.Parameter()
    files   = luigi.DictParameter()


    def output(self):
        dataset_path = Path(
            f"{TARGET_DIR}/datasets/{self.dataset}.{self.tag}.{self.source}"
        )
        metadata_path = dataset_path / "metadata.parquet"
        return luigi.LocalTarget(metadata_path, format=Nop)

    def _prepare_real(self, metadata: dict) -> pd.DataFrame:
        path = DATA_DIR / f"{self.dataset}/{self.tag}/raw"

        filepath = path / metadata["csv"]
        if not filepath.is_file():
            raise FileNotFoundError(f"File {filepath} not found.")
        data = pd.read_csv(filepath).rename(
            columns={"target": "label", "image_path": "path"}
        )
        data["name"] = self.dataset
        data["type"] = "real"
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

    def _prepare_pix2pix(self, metadata: dict) -> pd.DataFrame:
        path = DATA_DIR / f"{self.dataset}/{self.tag}/fake_images"
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
            data["name"] = self.dataset
            data["source"] = "pix2pix"
            metadata_list.append(data)

        return pd.concat(metadata_list)

    def _prepare_wgan(self, metadata: dict) -> pd.DataFrame:
        path = DATA_DIR / f"{self.dataset}/{self.tag}/fake_images"
        label_mapper = {"tb": True, "notb": False}

        metadata_list = []
        for label in metadata:
            filepath = path / metadata[label]
            if not filepath.is_file():
                raise FileNotFoundError(f"File {filepath} not found.")

            data = pd.read_csv(filepath, usecols=["image_path", "test", "sort"])
            data = data.sample(n=600, random_state=42)  # sample a fraction of images
            data.rename(
                columns={"test": "fold", "sort": "inner_fold", "image_path": "path"},
                inplace=True,
            )
            data["label"] = label_mapper[label]
            data["type"] = "fake"
            data["name"] = self.dataset
            data["source"] = "wgan"
            metadata_list.append(data)

        data_train, data_valid = train_test_split(
            pd.concat(metadata_list), test_size=0.2, shuffle=True, random_state=512
        )
        data_train["set"] = "train"
        data_valid["set"] = "val"

        return pd.concat([data_train, data_valid])

    def _prepare_cycle(self, metadata: dict) -> pd.DataFrame:
        path = DATA_DIR / f"{self.dataset}/{self.tag}/fake_images"
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
            data["name"] = self.dataset
            data["source"] = "cycle"
            metadata_list.append(data)

        return pd.concat(metadata_list)

    def run(self):
        if self.source == "raw":
            metadata = self._prepare_real(self.files)
        elif self.source == "pix2pix":
            metadata = self._prepare_pix2pix(self.files)
        elif self.source == "wgan":
            metadata = self._prepare_wgan(self.files)
        elif self.source == "cycle":
            metadata = self._prepare_cycle(self.files)
        else:
            raise KeyError(f"Source '{self.source}' is not defined.")

        with self.output().open("w") as f:
            metadata.to_parquet(f)
