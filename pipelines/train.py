import luigi
from tasks.convnets import (
    TrainAltogether,
    TrainBaseline,
    TrainBaselineFineTuning,
    TrainInterleaved,
    TrainSynthetic,
)

luigi.interface.core.log_level = "WARNING"


class BaselinePipeline(luigi.WrapperTask):
    dataset_info = luigi.DictParameter()
    batch_size = luigi.IntParameter()
    epochs = luigi.IntParameter()
    learning_rate = luigi.IntParameter()
    image_width = luigi.IntParameter()
    image_height = luigi.IntParameter()
    grayscale = luigi.BoolParameter()
    job = luigi.DictParameter()

    def requires(self):
        dataset_info = dict(sorted(self.dataset_info.items()))

        yield TrainBaseline(
            dataset_info=dataset_info,
            batch_size=self.batch_size,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            image_width=self.image_width,
            image_height=self.image_height,
            grayscale=self.grayscale,
            job=self.job,
        )


class SyntheticPipeline(luigi.WrapperTask):
    dataset_info = luigi.DictParameter()
    batch_size = luigi.IntParameter()
    epochs = luigi.IntParameter()
    learning_rate = luigi.IntParameter()
    image_width = luigi.IntParameter()
    image_height = luigi.IntParameter()
    grayscale = luigi.BoolParameter()
    job = luigi.DictParameter()

    def requires(self):
        dataset_info = dict(sorted(self.dataset_info.items()))

        yield TrainSynthetic(
            dataset_info=dataset_info,
            batch_size=self.batch_size,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            image_width=self.image_width,
            image_height=self.image_height,
            grayscale=self.grayscale,
            job=self.job,
        )


class InterleavedPipeline(luigi.WrapperTask):
    dataset_info = luigi.DictParameter()
    batch_size = luigi.IntParameter()
    epochs = luigi.IntParameter()
    learning_rate = luigi.IntParameter()
    image_width = luigi.IntParameter()
    image_height = luigi.IntParameter()
    grayscale = luigi.BoolParameter()
    job = luigi.DictParameter()

    def requires(self):
        yield TrainInterleaved(
            dataset_info=self.dataset_info,
            batch_size=self.batch_size,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            image_width=self.image_width,
            image_height=self.image_height,
            grayscale=self.grayscale,
            job=self.job,
        )


class AltogetherPipeline(luigi.WrapperTask):
    dataset_info = luigi.DictParameter()
    batch_size = luigi.IntParameter()
    epochs = luigi.IntParameter()
    learning_rate = luigi.IntParameter()
    image_width = luigi.IntParameter()
    image_height = luigi.IntParameter()
    grayscale = luigi.BoolParameter()
    job = luigi.DictParameter()

    def requires(self):
        yield TrainAltogether(
            dataset_info=self.dataset_info,
            batch_size=self.batch_size,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            image_width=self.image_width,
            image_height=self.image_height,
            grayscale=self.grayscale,
            job=self.job,
        )


class BaselineFineTuningPipeline(luigi.WrapperTask):
    dataset_info = luigi.DictParameter()
    batch_size = luigi.IntParameter()
    epochs = luigi.IntParameter()
    learning_rate = luigi.IntParameter()
    image_width = luigi.IntParameter()
    image_height = luigi.IntParameter()
    grayscale = luigi.BoolParameter()
    job = luigi.DictParameter()

    def requires(self):
        yield TrainBaselineFineTuning(
            dataset_info=self.dataset_info,
            batch_size=self.batch_size,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            image_width=self.image_width,
            image_height=self.image_height,
            grayscale=self.grayscale,
            job=self.job,
        )



class TaskCnvPipeline(luigi.WrapperTask):
    experiment_path = luigi.Parameter()
    output_path     = luigi.Parameter()

    def requires(self):
        yield TaskCnv(
            experiment_path = self.experiment_path, 
            output_path     = self.output_path
        )


processes = {
            "baseline"              : BaselinePipeline,
            "synthetic"             : SyntheticPipeline,
            "interleaved"           : InterleavedPipeline,
            "altogether"            : AltogetherPipeline,
            "baseline_fine_tuning"  : BaselineFineTuningPipeline,
        }