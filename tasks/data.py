import os
import luigi
from luigi.format import Nop
from dotenv import load_dotenv
from pathlib import Path
from tasks.commons import Task
from utils.data import prepare_data


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


    def run(self):
        metadata = prepare_data( self.source, DATA_DIR, self.dataset, self.tag, self.files)
        with self.output().open("w") as f:
            metadata.to_parquet(f)

