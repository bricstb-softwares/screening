from __future__ import annotations

import pickle
import luigi
import numpy as np
import pandas as pd
import os

from utils import commons
from datetime import timedelta
from itertools import product
from timeit import default_timer
from pprint import pprint
from pathlib import Path
from loguru import logger
from luigi import Task as LuigiTask
from tasks.data import CrossValidation
from utils.convnets import (
    save_job_state,
    evaluate_tuning,
    split_dataframe,
    build_model_from_train_state,
    prepare_model

)



class MyTask(LuigiTask):

    experiment_path   = luigi.Parameter()
    output_path       = luigi.Parameter()


    def set_logger(self):
        commons.create_folder(self.output_path)
        commons.set_task_logger(
            log_prefix=f"{self.__class__.__name__}", log_path=self.output_path
        )

    def get_output_path(self):   
        return self.output_path

    def requires(self) -> list[CrossValidation]:

        task_params = self.get_task_params_from( self.experiment_path )
        # get dataset info from the current task
        dataset_info = task_params['dataset_info']

        required_tasks = []
        for dataset in dataset_info:
            for source in dataset_info[dataset]["sources"]:
                required_tasks.append(
                    CrossValidation(
                        dataset,
                        dataset_info[dataset]["tag"],
                        source,
                        dataset_info[dataset]["sources"][source],
                    )
                )

        return required_tasks


    def output(self) -> luigi.LocalTarget:
        output_file = Path(self.output_path) / "task_params.pkl" 
        return luigi.LocalTarget(str(output_file))


    def get_data(self, tasks):
        data_list = []
        for task in tasks:
            if type(task) == CrossValidation:
                data_list.append(pd.read_parquet(task.output().path))
        data = pd.concat(data_list)
        data = data.sample(frac=1, random_state=42)
        return data


    def get_task_params_from( self, experiment_path):
        params = pickle.load(open(experiment_path+'/task_params.pkl', 'rb'))
        return params
 

    def get_train_state_from(self, test, sort):
        model_path   = self.experiment_path +f"/cnn_fold{test}/sort{sort}/"
        model, history, params = build_model_from_train_state( model_path )
        return prepare_model(model, history, params)

    def get_hash(self):
        task_params = self.get_task_params_from(self.experiment_path)



class Baseline(MyTask):

    def run(self):

        self.set_logger()
        logger.info(f"Running {self.get_task_family()}...")
    
        # reading all parameters from the current task
        task_params     = self.get_task_params_from( self.experiment_path )
        # get training control tags
        experiment_type = task_params['experiment_id'].split('_')[0]
        experiment_hash = task_params['experiment_id'].split('_')[1]

        pprint(task_params)
        # load the dataset
        tasks           = self.requires()
        # get the data
        data            = self.get_data(tasks)

        start = default_timer()
        for i, j in product( range(10), range(9) ):

            folder_path = f'{self.output_path}/job.test_{i}.sort_{j}'
            commons.create_folder(folder_path)
            job_path = folder_path + '/output.pkl'
            if os.path.exists(job_path):
                continue
            
            logger.info(f"Converting Fold #{i} / Validation #{j}...")
            train_real  = split_dataframe(data, i, j, "train_real")
            valid_real  = split_dataframe(data, i, j, "valid_real")
            test_real   = split_dataframe(data, i, j, "test_real" )

            # get train state
            train_state = self.get_train_state_from(i,j)
            train_state = evaluate_tuning(train_state, train_real, valid_real, test_real, task_params)
       
            logger.info(f"Saving job state for test {i} and sort {j} into {job_path}")
            save_job_state( job_path,
                            train_state, 
                            job_params      = {'test':i, 'sort':j}, 
                            task_params     = task_params,
                            experiment_hash = experiment_hash,
                            experiment_type = experiment_type,
                            )

        end = default_timer()

        # output results
        with open(self.output().path, "wb") as file:
             pickle.dump(task_params, file, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info("")
        logger.info(f" - Output: {self.get_output_path()}")
        logger.info("")
        logger.info(f"=== End: '{self.__class__.__name__}' ===")
        logger.info("")

