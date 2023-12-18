from __future__ import annotations

import logging
import pickle
from collections import defaultdict
from datetime import timedelta
from itertools import product
from pathlib import Path
from timeit import default_timer

import luigi
import numpy as np
import pandas as pd
from tasks.commons import Task
from tasks.data import CrossValidation
from utils.convnets import (
    save_train_state,
    save_job_state,
    split_dataframe,
    train_altogether,
    train_fine_tuning,
    train_interleaved,
    train_neural_net,
)


class TrainCNN(Task):
    dataset_info = luigi.DictParameter()
    batch_size = luigi.IntParameter()
    epochs = luigi.IntParameter()
    learning_rate = luigi.IntParameter()
    image_width = luigi.IntParameter()
    image_height = luigi.IntParameter()
    grayscale = luigi.BoolParameter()
    job = luigi.DictParameter(default={}, significant=False)


    def log_params(self, dirname=''):

        self.set_logger()

        logging.info(f"=== Start '{self.__class__.__name__}' ===\n")
        logging.info("Dataset Info:")
        task_params = self.__dict__["param_kwargs"].copy()

        for dataset in task_params["dataset_info"]:
            tag = task_params["dataset_info"][dataset]["tag"]
            sources = sorted(task_params["dataset_info"][dataset]["sources"].keys())
            logging.info(f"{dataset}")
            logging.info(f" - tag: {tag}")
            logging.info(f" - sources: {sources}")
        logging.info("\n")

        logging.info("Training Parameters:")
        for key in task_params:
            if key == "dataset_info":
                continue
            logging.info(f" - {key}: {task_params[key]}")
        logging.info("")

        return task_params

    def requires(self) -> list[CrossValidation]:
        required_tasks = []
        for dataset in self.dataset_info:
            for source in self.dataset_info[dataset]["sources"]:
                required_tasks.append(
                    CrossValidation(
                        dataset,
                        self.dataset_info[dataset]["tag"],
                        source,
                        self.dataset_info[dataset]["sources"][source],
                    )
                )

        return required_tasks

    def output(self) -> luigi.LocalTarget:
        file_name = "output.pkl" if self.get_job_params() else "task_params.pkl"
        output_file = Path(self.get_output_path()) / file_name
        return luigi.LocalTarget(str(output_file))




class TrainBaseline(TrainCNN):
    def run(self):
        task_params = self.log_params()

        metadata_list = []
        for i in range(len(self.requires())):
            cross_val_path = self.requires()[i].output().path
            metadata_list.append(pd.read_parquet(cross_val_path))

        metadata = pd.concat(metadata_list)
        metadata = metadata.sample(frac=1, random_state=42)

        # training loop
        job_params = self.get_job_params()  
        folds = list(range(10)) if not job_params else [job_params['test']]
        inner_folds = list(range(9)) if not job_params else [job_params['sort']]
        experiment_path = Path(self.get_output_path())

        start = default_timer()
        for i, j in product(folds, inner_folds):
            logging.info(f"Fold #{i} / Validation #{j}")
            train_real = split_dataframe(metadata, i, j, "train_real")
            valid_real = split_dataframe(metadata, i, j, "valid_real")
            train_state = train_neural_net(train_real, valid_real, task_params)
            output_path = experiment_path/f"fold{i}/sort{j}/" if not job_params else experiment_path
            save_train_state(output_path, train_state)
        end = default_timer()

        # output results
        task_params["experiment_id"] = experiment_path.name
        task_params["training_time"] = timedelta(seconds=(end - start))

        if job_params: # as job
            logging.info(f"Saving job state for test {i} and sort {j} into {self.output().path}")
            save_job_state( self.output().path, metadata,
                            train_state, 
                            job_params      = job_params, 
                            task_params     = task_params,
                            dataset_info    = self.dataset_info,
                            hash_experiment = self.get_hash() )

        else: # as single task
            with open(self.output().path, "wb") as file:
                pickle.dump(task_params, file, protocol=pickle.HIGHEST_PROTOCOL)

        logging.info("")
        logging.info(f" - Output: {self.get_output_path()}")
        logging.info("")
        logging.info(f"=== End: '{self.__class__.__name__}' ===")
        logging.info("")


class TrainSynthetic(TrainCNN):
    def run(self):
        task_params = self.log_params()

        metadata_list = []
        for i in range(len(self.requires())):
            cross_val_path = self.requires()[i].output().path
            metadata_list.append(pd.read_parquet(cross_val_path))

        metadata = pd.concat(metadata_list)
        metadata = metadata.sample(frac=1, random_state=42)
        job      = task_params.get('job', None)    

        # training loop
        folds = list(range(10)) if not job else [job['test']]
        inner_folds = list(range(9)) if not job else [job['sort']]
        experiment_path = Path(self.get_output_path())

        start = default_timer()
        for i, j in product(folds, inner_folds):
            logging.info(f"Fold #{i} / Validation #{j}")
            train_fake = split_dataframe(metadata, i, j, "train_fake")
            valid_real = split_dataframe(metadata, i, j, "valid_real")
            train_state = train_neural_net(train_fake, valid_real, task_params)
            output_path = experiment_path / f"cnn_fold{i}/sort{j}/"
            save_train_state(output_path, train_state)
        end = default_timer()

        # output results
        task_params["experiment_id"] = experiment_path.name
        task_params["training_time"] = timedelta(seconds=(end - start))
        with open(self.output().path, "wb") as file:
            pickle.dump(task_params, file, protocol=pickle.HIGHEST_PROTOCOL)

        logging.info("")
        logging.info(f" - Output: {self.get_output_path()}")
        logging.info("")
        logging.info(f"=== End: '{self.__class__.__name__}' ===")
        logging.info("")


class TrainInterleaved(TrainCNN):
    def run(self):
        task_params = self.log_params()

        metadata_list = []
        for i in range(len(self.requires())):
            cross_val_path = self.requires()[i].output().path
            metadata_list.append(pd.read_parquet(cross_val_path))

        metadata = pd.concat(metadata_list)
        metadata = metadata.sample(frac=1, random_state=42)
        job      = task_params.get('job', None)    

        # training loop
        folds = list(range(10)) if not job else [job['test']]
        inner_folds = list(range(9)) if not job else [job['sort']]
        experiment_path = Path(self.get_output_path())

        start = default_timer()
        for i, j in product(folds, inner_folds):
            logging.info(f"Fold #{i} / Validation #{j}")
            train_fake = split_dataframe(metadata, i, j, "train_fake")
            train_real = split_dataframe(metadata, i, j, "train_real")
            valid_real = split_dataframe(metadata, i, j, "valid_real")

            train_state = train_interleaved(
                train_real, train_fake, valid_real, task_params
            )
            output_path = experiment_path / f"cnn_fold{i}/sort{j}/"
            save_train_state(output_path, train_state)
        end = default_timer()

        task_params["experiment_id"] = experiment_path.name
        task_params["training_time"] = timedelta(seconds=(end - start))
        with open(self.output().path, "wb") as file:
            pickle.dump(task_params, file, protocol=pickle.HIGHEST_PROTOCOL)

        logging.info("")
        logging.info(f" - Output: {self.get_output_path()}")
        logging.info("")
        logging.info(f"=== End: '{self.__class__.__name__}' ===")
        logging.info("")


class TrainAltogether(TrainCNN):
    def run(self):
        task_params = self.log_params()

        metadata_list = []
        for i in range(len(self.requires())):
            cross_val_path = self.requires()[i].output().path
            metadata_list.append(pd.read_parquet(cross_val_path))

        metadata = pd.concat(metadata_list)
        metadata = metadata.sample(frac=1, random_state=42)
        job      = task_params.get('job', None)    

        # training loop
        folds = list(range(10)) if not job else [job['test']]
        inner_folds = list(range(9)) if not job else [job['sort']]
        experiment_path = Path(self.get_output_path())

        start = default_timer()
        for i, j in product(folds, inner_folds):
            logging.info(f"Fold #{i} / Validation #{j}")
            train_real = split_dataframe(metadata, i, j, "train_real")
            train_fake = split_dataframe(metadata, i, j, "train_fake")
            valid_real = split_dataframe(metadata, i, j, "valid_real")

            real_weights = (1 / len(train_real)) * np.ones(len(train_real))
            fake_weights = (1 / len(train_fake)) * np.ones(len(train_fake))
            weights = np.concatenate((real_weights, fake_weights))
            weights = weights / sum(weights)

            train_state = train_altogether(
                train_real, train_fake, valid_real, weights, task_params
            )
            output_path = experiment_path / f"cnn_fold{i}/sort{j}/"
            save_train_state(output_path, train_state)
        end = default_timer()

        task_params["experiment_id"] = experiment_path.name
        task_params["training_time"] = timedelta(seconds=(end - start))
        with open(self.output().path, "wb") as file:
            pickle.dump(task_params, file, protocol=pickle.HIGHEST_PROTOCOL)

        logging.info("")
        logging.info(f" - Output: {self.get_output_path()}")
        logging.info("")
        logging.info(f"=== End: '{self.__class__.__name__}' ===")
        logging.info("")


class TrainBaselineFineTuning(TrainCNN):
    def requires(self) -> list:
        baseline_info = defaultdict(dict)
        for dataset in self.dataset_info:
            baseline_info[dataset]["tag"] = self.dataset_info[dataset]["tag"]
            sources = self.dataset_info[dataset]["sources"]
            sources = {k: v for k, v in sources.items() if k == "raw"}
            baseline_info[dataset]["sources"] = sources

        baseline_train = TrainBaseline(
            dataset_info=baseline_info,
            batch_size=self.batch_size,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            image_width=self.image_width,
            image_height=self.image_height,
            grayscale=self.grayscale,
        )

        required_tasks = [baseline_train]
        for dataset in self.dataset_info:
            for source in self.dataset_info[dataset]["sources"]:
                required_tasks.append(
                    CrossValidation(
                        dataset,
                        self.dataset_info[dataset]["tag"],
                        source,
                        self.dataset_info[dataset]["sources"][source],
                    )
                )

        return required_tasks

    def run(self):
        task_params = self.log_params()

        required_params = self.requires()[0].param_args
        baseline_task = TrainBaseline(*required_params)
        baseline_path = Path(baseline_task.get_output_path())
        logging.info(f"Baseline Experiment: {baseline_path}\n")

        metadata_list = []
        for i in range(len(self.requires())):
            if i == 0:
                continue
            cross_val_path = self.requires()[i].output().path
            metadata_list.append(pd.read_parquet(cross_val_path))

        metadata = pd.concat(metadata_list)
        metadata = metadata.sample(frac=1, random_state=42)
        job      = task_params.get('job', None)    

        # training loop
        folds = list(range(10)) if not job else [job['test']]
        inner_folds = list(range(9)) if not job else [job['sort']]
        experiment_path = Path(self.get_output_path())

        start = default_timer()
        for i, j in product(folds, inner_folds):
            logging.info(f"Fold #{i} / Validation #{j}")
            model_path = baseline_path / f"cnn_fold{i}/sort{j}/"
            train_fake = split_dataframe(metadata, i, j, "train_fake")
            valid_real = split_dataframe(metadata, i, j, "valid_real")
            train_state = train_fine_tuning(
                train_fake, valid_real, task_params, model_path
            )
            output_path = experiment_path / f"cnn_fold{i}/sort{j}/"
            save_train_state(output_path, train_state)
        end = default_timer()

        # output results
        task_params["experiment_id"] = experiment_path.name
        task_params["training_time"] = timedelta(seconds=(end - start))
        with open(self.output().path, "wb") as file:
            pickle.dump(task_params, file, protocol=pickle.HIGHEST_PROTOCOL)

        logging.info("")
        logging.info(f" - Output: {self.get_output_path()}")
        logging.info("")
        logging.info(f"=== End: '{self.__class__.__name__}' ===")
        logging.info("")


class TrainFineTuning(TrainCNN):
    def requires(self) -> list[TrainSynthetic]:
        synthetic_train = TrainSynthetic(
            dataset_info=self.dataset_info,
            batch_size=self.batch_size,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            image_width=self.image_width,
            image_height=self.image_height,
            grayscale=self.grayscale,
        )

        return [synthetic_train]

    def run(self):
        task_params = self.log_params()

        synthetic_task = TrainSynthetic(
            dataset_info=self.dataset_info,
            batch_size=self.batch_size,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            image_width=self.image_width,
            image_height=self.image_height,
            grayscale=self.grayscale,
        )
        synthetic_path = Path(synthetic_task.get_output_path())

        metadata_list = []
        for i in range(len(self.requires())):
            metadata_list.append(pd.read_parquet(self.requires()[i].output().path))

        metadata = pd.concat(metadata_list)
        metadata = metadata.sample(frac=1, random_state=42)
        job      = task_params.get('job', None)    

        # training loop
        folds = list(range(10)) if not job else [job['test']]
        inner_folds = list(range(9)) if not job else [job['sort']]
        experiment_path = Path(self.get_output_path())

        start = default_timer()
        for i, j in product(folds, inner_folds):
            logging.info(f"Fold #{i} / Validation #{j}")
            model_path = synthetic_path / f"cnn_fold{i}/sort{j}/"
            train_real = split_dataframe(metadata, i, j, "train_real")
            valid_real = split_dataframe(metadata, i, j, "valid_real")
            train_state = train_fine_tuning(
                train_real, valid_real, task_params, model_path
            )
            output_path = experiment_path / f"cnn_fold{i}/sort{j}/"
            save_train_state(output_path, train_state)
        end = default_timer()

        # output results
        task_params["experiment_id"] = experiment_path.name
        task_params["training_time"] = timedelta(seconds=(end - start))
        with open(self.output().path, "wb") as file:
            pickle.dump(task_params, file, protocol=pickle.HIGHEST_PROTOCOL)

        logging.info("")
        logging.info(f" - Output: {self.get_output_path()}")
        logging.info("")
        logging.info(f"=== End: '{self.__class__.__name__}' ===")
        logging.info("")
