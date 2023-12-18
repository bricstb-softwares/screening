import os
from pathlib import Path

import luigi
from tasks.report import ReportCNN

luigi.interface.core.log_level = "WARNING"
DATA_DIR = Path(os.environ['DATA_DIR'])
TARGET_DIR = Path(os.environ['TARGET_DIR'])

class ReportPipeline(luigi.WrapperTask):
    experiment_hash = luigi.Parameter()
    dataset_info = luigi.DictParameter()

    def requires(self):
        yield ReportCNN(
            experiment_hash=self.experiment_hash,
            dataset_info=self.dataset_info,
        )
