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
    max_inter     = luigi.IntParameter()
    image_width   = luigi.IntParameter()
    image_height  = luigi.IntParameter()
    grayscale     = luigi.BoolParameter()
    job           = luigi.DictParameter(default={}, significant=False)


  

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
                output_path = experiment_path/f"svm_fold{test}/sort{sort}/" 
                save_train_state(output_path, train_state)

        end = default_timer()

        # output results
        task_params["experiment_id"] = experiment_path.name
        task_params["training_time"] = timedelta(seconds=(end - start))

        if not job_params:
            with open(self.output().path, "wb") as file:
                pickle.dump(task_params, file, protocol=pickle.HIGHEST_PROTOCOL)




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