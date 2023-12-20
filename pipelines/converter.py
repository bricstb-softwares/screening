

import luigi
from tasks.converter import Baseline

class BaselinePipeline(luigi.WrapperTask):
    experiment_path = luigi.Parameter()
    output_path     = luigi.Parameter()

    def requires(self):
        yield Baseline(
            experiment_path = self.experiment_path, 
            output_path     = self.output_path
        )