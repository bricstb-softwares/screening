import os
import luigi
import pandas as pd
from sklearn.model_selection import train_test_split
from itertools import product
from luigi.format import Nop
from dotenv import load_dotenv
from pathlib import Path
from tasks.commons import Task


from utils.data import (
    prepare_real,
    prepare_p2p,
    prepare_wgan,
    _prepare_cycle
)

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

   def prepare_cycle(data_dir : str, dataset : str, tag : str, metadata: dict) -> pd.DataFrame:

    def run(self):
        if self.source == "raw":
            metadata = prepare_real(DATA_DIR, self.dataset, self.tag, self.files)
        elif self.source == "pix2pix":
            metadata = prepare_p2p(DATA_DIR, self.dataset, self.tag, self.files)
        elif self.source == "wgan":
            metadata = prepare_wgan(DATA_DIR, self.dataset, self.tag, self.files)
        elif self.source == "cycle":
            metadata = prepare_cycle(DATA_DIR, self.dataset, self.tag, self.files)
        else:
            raise KeyError(f"Source '{self.source}' is not defined.")

        with self.output().open("w") as f:
            metadata.to_parquet(f)
