__all__ = ["ReportPipeline"]

import os
import luigi

from pathlib import Path
from screening.tasks import ReportCNN
from screening import TARGET_DIR, DATA_DIR

luigi.interface.core.log_level = "WARNING"


class ReportPipeline(luigi.WrapperTask):
    experiment_hash = luigi.Parameter()
    dataset_info = luigi.DictParameter()

    def requires(self):
        yield ReportCNN(
            experiment_hash=self.experiment_hash,
            dataset_info=self.dataset_info,
        )
