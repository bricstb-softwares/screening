from __future__ import annotations

__all__ = []



import pickle
import luigi
import numpy as np
import pandas as pd

from collections import defaultdict
from datetime import timedelta
from itertools import product
from pathlib import Path
from timeit import default_timer
from loguru import logger


from screening.tasks import Task, CrossValidation
from screening.utils.svm import (
    save_train_state,
    save_job_state,
    build_model_from_job,
    build_model_from_train_state,
    train_svm,
)
from screening.utils.data import (
    split_dataframe,
)
from screening.validation import (
    evaluate
)


#
# Base SVM class
#
class TrainSVM(Task):

    dataset_info  = luigi.DictParameter()
    epochs        = luigi.IntParameter()
    image_width   = luigi.IntParameter()
    image_height  = luigi.IntParameter()
    grayscale     = luigi.BoolParameter()
    job           = luigi.DictParameter(default={}, significant=False)


  
    def output(self) -> luigi.LocalTarget:
        file_name = "output.pkl" if self.get_job_params() else "task_params.pkl"
        output_file = Path(self.get_output_path()) / file_name
        return luigi.LocalTarget(str(output_file))


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

    #
    # Run task!
    #
    def run(self):
        
        task_params     = self.log_params()
        logger.info(f"Running {self.get_task_family()}...")

        tasks           = self.requires()
        data            = self.get_data_samples(tasks)
        job_params      = self.get_job_params()  

        experiment_path = Path(self.get_output_path())

        start = default_timer()

        for test, sort in product(self.get_tests(), self.get_sorts()):

            logger.info(f"Fold #{test} / Validation #{sort}")
            train_state = self.fit( data, test, sort, task_params )

            if job_params:
                logger.info(f"Saving job state for test {test} and sort {sort} into {self.output().path}")
                save_job_state( self.output().path,
                        train_state, 
                        test = test,
                        sort = sort,
                        metadata = {
                            "hash"        : self.get_hash(),
                            "type"        : self.get_task_family(),
                            "task_params" : task_params,
                        }
                    )
            else:
                output_path = experiment_path/f"cnn_fold{test}/sort{sort}/" 
                save_train_state(output_path, train_state)

        end = default_timer()

        # output results
        task_params["experiment_id"] = experiment_path.name
        task_params["training_time"] = timedelta(seconds=(end - start))

        if not job_params:
            with open(self.output().path, "wb") as file:
                pickle.dump(task_params, file, protocol=pickle.HIGHEST_PROTOCOL)




    def log_params(self):
        self.set_logger()

        logger.info(f"=== Start '{self.__class__.__name__}' ===\n")
        logger.info("Dataset Info:")
        task_params = self.__dict__["param_kwargs"].copy()

        for dataset in task_params["dataset_info"]:
            tag = task_params["dataset_info"][dataset]["tag"]
            sources = sorted(task_params["dataset_info"][dataset]["sources"].keys())
            logger.info(f"{dataset}")
            logger.info(f" - tag: {tag}")
            logger.info(f" - sources: {sources}")
        logger.info("\n")

        logger.info("Training Parameters:")
        for key in task_params:
            if key == "dataset_info":
                continue
            logger.info(f" - {key}: {task_params[key]}")
        logger.info("")
        
        logger.info(f"Experiment hash: {self.get_hash()}")
        logger.info("")

        return task_params

    def get_data_samples(self, tasks, seed : int=42):
        data_list = []
        for task in tasks:
            if type(task) == CrossValidation:
                data_list.append(pd.read_parquet(task.output().path))
        data = pd.concat(data_list)
        data = data.sample(frac=1, random_state=seed)
        return data


    def get_sorts(self):
        job_params = self.get_job_params()  
        return list(range(9)) if not job_params else [job_params['sort']]

    def get_tests(self):
        job_params = self.get_job_params()  
        return list(range(10)) if not job_params else [job_params['test']]





#
# Train methods
#

class TrainBaseline(TrainSVM):
  
    def fit(self, data, test, sort, task_params ):

        start = default_timer()

        train_real  = split_dataframe(data, test, sort, "train_real")
        valid_real  = split_dataframe(data, test, sort, "valid_real")
        test_real   = split_dataframe(data, test, sort, "test_real" )

        train_state = train_svm(train_real, valid_real, task_params)
        train_state = evaluate( train_state, train_real, valid_real, test_real)

        end = default_timer()
        logger.info(f"training toke {timedelta(seconds=(end - start))}...")
        return train_state

       
class TrainSynthetic(TrainCNN):

    def fit(self, data, test, sort, task_params ):

        start = default_timer()

        train_fake  = split_dataframe(data, test, sort, "train_fake")
        valid_real  = split_dataframe(data, test, sort, "valid_real")
        test_real   = split_dataframe(data, test, sort, "test_real" )

        train_state = train_svm(train_fake, valid_real, task_params)
        train_state = evaluate( train_state, train_fake, valid_real, test_real)

        end = default_timer()
        logger.info(f"training toke {timedelta(seconds=(end - start))}...")

        return train_state