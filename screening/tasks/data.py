from __future__ import annotations

__all__ = ["CrossValidation"]

import luigi
from luigi.format import Nop
from pathlib import Path

from screening import prepare_data, TARGET_DIR
from screening.tasks.commons import Task


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
        metadata = prepare_data( self.source, self.dataset, self.tag, self.files)
        with self.output().open("w") as f:
            metadata.to_parquet(f)

