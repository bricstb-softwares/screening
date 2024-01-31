__all__ = []

import luigi
from screening.tasks.svm import (
    TrainBaseline,
    TrainSynthetic,
)

luigi.interface.core.log_level = "WARNING"


class BaselinePipeline(luigi.WrapperTask):

    dataset_info  = luigi.DictParameter()
    max_int       = luigi.IntParameter()
    image_width   = luigi.IntParameter()
    image_height  = luigi.IntParameter()
    grayscale     = luigi.BoolParameter()
    job           = luigi.DictParameter()

    def requires(self):
        dataset_info = dict(sorted(self.dataset_info.items()))

        yield TrainBaseline(
            dataset_info = dataset_info,
            max_int      = self.max_int,
            image_width  = self.image_width,
            image_height = self.image_height,
            grayscale    = self.grayscale,
            job          = self.job,
        )



class SyntheticPipeline(luigi.WrapperTask):

    dataset_info  = luigi.DictParameter()
    max_int       = luigi.IntParameter()
    image_width   = luigi.IntParameter()
    image_height  = luigi.IntParameter()
    grayscale     = luigi.BoolParameter()
    job           = luigi.DictParameter()

    def requires(self):
        dataset_info = dict(sorted(self.dataset_info.items()))

        yield TrainSynthetic(
            dataset_info = dataset_info,
            max_int      = self.max_int,
            image_width  = self.image_width,
            image_height = self.image_height,
            grayscale    = self.grayscale,
            job          = self.job,
        )



processes = {
            "baseline"              : BaselinePipeline,
            "synthetic"             : SyntheticPipeline,
        }