from __future__ import annotations

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
from tasks.commons import Task
from tasks.data import CrossValidation
from utils.convnets import (
    save_train_state,
    save_job_state,
    build_model_from_job,
    build_model_from_train_state,
    split_dataframe,
    train_altogether,
    train_fine_tuning,
    train_interleaved,
    train_neural_net,
    evaluate_tuning,
)


#
# Base ConvNet class
#
class TrainCNN(Task):

    dataset_info  = luigi.DictParameter()
    batch_size    = luigi.IntParameter()
    epochs        = luigi.IntParameter()
    learning_rate = luigi.IntParameter()
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
                        job_params      = job_params, 
                        task_params     = task_params,
                        experiment_hash = self.get_hash(),
                        experiment_type = self.get_task_family(),
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

class TrainBaseline(TrainCNN):
  
    def fit(self, data, test, sort, task_params ):

        start = default_timer()

        train_real  = split_dataframe(data, test, sort, "train_real")
        valid_real  = split_dataframe(data, test, sort, "valid_real")
        test_real   = split_dataframe(data, test, sort, "test_real" )

        train_state = train_neural_net(train_real, valid_real, task_params)
        train_state = evaluate_tuning(train_state, train_real, valid_real, test_real, task_params)

        end = default_timer()
        train_state.timer = timedelta(seconds=(end - start))
        return train_state

       
class TrainSynthetic(TrainCNN):

    def fit(self, data, test, sort, task_params ):

        start = default_timer()

        train_real  = split_dataframe(data, test, sort, "train_fake")
        valid_real  = split_dataframe(data, test, sort, "valid_real")
        test_real   = split_dataframe(data, test, sort, "test_real" )

        train_state = train_neural_net(train_real, valid_real, task_params)
        train_state = evaluate_tuning(train_state, train_real, valid_real, test_real, task_params)

        end = default_timer()
        train_state.timer = timedelta(seconds=(end - start))

        return train_state


class TrainInterleaved(TrainCNN):

    def fit(self, data, test, sort, task_params ):
        start = default_timer()

        train_fake = split_dataframe(data, i, j, "train_fake")
        train_real = split_dataframe(data, i, j, "train_real")
        valid_real = split_dataframe(data, i, j, "valid_real")
        test_real  = split_dataframe(data, i, j, "test_real" )
        train_state = train_interleaved(
            train_real, train_fake, valid_real, task_params
        )
        train_real_fake = pd.concat([train_real, train_fake])
        train_state = evaluate_tuning(train_state, train_real_fake, valid_real, test_real, task_params)

        end = default_timer()
        train_state.timer = timedelta(seconds=(end - start))

        return train_state

        
class TrainAltogether(TrainCNN):


    def fit(self, data, test, sort, task_params ):

        start = default_timer()

        train_real = split_dataframe(data, i, j, "train_real")
        train_fake = split_dataframe(data, i, j, "train_fake")
        valid_real = split_dataframe(data, i, j, "valid_real")
        test_real  = split_dataframe(data, i, j, "test_real" )
        real_weights = (1 / len(train_real)) * np.ones(len(train_real))
        fake_weights = (1 / len(train_fake)) * np.ones(len(train_fake))
        weights = np.concatenate((real_weights, fake_weights))
        weights = weights / sum(weights)
        train_state = train_altogether(
            train_real, train_fake, valid_real, weights, task_params
        )
        train_real_fake = pd.concat([train_real, train_fake])
        train_state = evaluate_tuning(train_state, train_real_fake, valid_real, test_real, task_params)
        end = default_timer()
        train_state.timer = timedelta(seconds=(end - start))

        return train_state
            


class TrainBaselineFineTuning(TrainCNN):


    def get_parent_model(self, test, sort):
           
        job_params = self.get_job_params()  
        if job_params and job_params['parent']:
            model_path = job_params['parent'] + f'/job.test_{test}.sort_{sort}'
            model, _, _     = build_model_from_job( model_path+'/output.pkl' )
        else:
            tasks           = self.requires()
            experiement_path= Path(tasks[0].get_output_path())
            model_path      = experiment_path / f"cnn_fold{test}/sort{sort}/"
            model, _, _     = build_model_from_train_state( model_path )
        return model


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
            job_params=self.job_params,
        )

        required_tasks = [baseline_train]
        required_tasks.update( TrainCNN.requires() )
        return required_tasks


    def fit(self, data, test, sort, task_params ):
        
        start = default_timer()

        train_fake = split_dataframe(data, test, sort, "train_fake")
        valid_real = split_dataframe(data, test, sort, "valid_real")
        test_real  = split_dataframe(data, test, sort, "test_real" )
        model      = self.get_parent_model(test, sort)

        train_state = train_fine_tuning(
            train_fake, valid_real, test_real, task_params, model
        )
        train_state = evaluate_tuning(train_state, train_fake, valid_real, test_real, task_params)

        end = default_timer()
        train_state.timer = timedelta(seconds=(end - start))

        return train_state
       


class TrainFineTuning(TrainCNN):


    def get_parent_model(self, test, sort):
           
        job_params = self.get_job_params()  
        if job_params and job_params['parent']:
            model_path = job_params['parent'] + f'/job.test_{test}.sort_{sort}'
            model, _, _     = build_model_from_job( model_path+'/output.pkl' )
        else:
            tasks           = self.requires()
            experiement_path= Path(tasks[0].get_output_path())
            model_path      = experiment_path / f"cnn_fold{test}/sort{sort}/"
            model, _, _     = build_model_from_train_state( model_path )

        return model


    def requires(self) -> list[TrainSynthetic]:
        synthetic_train = TrainSynthetic(
            dataset_info=self.dataset_info,
            batch_size=self.batch_size,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            image_width=self.image_width,
            image_height=self.image_height,
            grayscale=self.grayscale,
            job_params=self.job_params,
        )
        required_tasks = [synthetic_train]
        required_tasks.update( TrainCNN.requires() )
        return required_tasks


    def fit(self, data, test, sort, task_params ):
        
        start = default_timer()

        train_real = split_dataframe(data, i, j, "train_real")
        valid_real = split_dataframe(data, i, j, "valid_real")
        test_real  = split_dataframe(data, i, j, "test_real" )
        model      = self.get_parent_model(i,j)
                    
        train_state = train_fine_tuning(
                    train_real, valid_real, task_params, model
                )
        train_state = evaluate_tuning(train_state, train_real, valid_real, test_real, task_params)
       
        end = default_timer()
        train_state.timer = timedelta(seconds=(end - start))

        return train_state




#
# 
#
class Converter(Task):

    experiment_path   = luigi.Parameter()
    output_path       = luigi.Parameter()


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


    def get_data_samples(self, tasks, seed=42):
        data_list = []
        for task in tasks:
            if type(task) == CrossValidation:
                data_list.append(pd.read_parquet(task.output().path))
        data = pd.concat(data_list)
        data = data.sample(frac=1, random_state=seed)
        return data


    def get_task_params_from( self, experiment_path):
        params = pickle.load(open(experiment_path+'/task_params.pkl', 'rb'))
        return params
 

    def get_train_state_from(self, test, sort):
        model_path   = self.experiment_path +f"/cnn_fold{test}/sort{sort}/"
        model, history, params = build_model_from_train_state( model_path )
        return prepare_model(model, history, params)

    

    def run(self):

        self.set_logger()
        logger.info(f"Running {self.get_task_family()}...")
    
        # reading all parameters from the current task
        task_params     = self.get_task_params_from( self.experiment_path )
        tasks           = self.requires()
        data            = self.get_data_samples(tasks)
        # get training control tags
        experiment_type = task_params['experiment_id'].split('_')[0].lower().replace('train','')
        experiment_hash = task_params['experiment_id'].split('_')[1]

        logger.info(f"Converting experiment with hash {experiment_hash}")

        start = default_timer()
        for test, sort in product( range(10), range(9) ):

            logger.info(f"Converting Fold #{test} / Validation #{sort}...")

            folder_path = f'{self.output_path}/job.test_{test}.sort_{sort}'
            commons.create_folder(folder_path)
            job_path = folder_path + '/output.pkl'
            if os.path.exists(job_path):
                logger.warning(f'job output exist into {folder_path}')
                continue
            
            logger.info(f"Experiment type is {experiment_type}")

            if experiment_type in ['baseline', 'finetuning']:
                train_data  = split_dataframe(data, test, sort, "train_real")
            elif experiment_type in ['synthetic', 'baselinefinetuning']:
                train_data  = split_dataframe(data, test, sort, "train_fake")
            elif experiment_type in ['interleaved', 'altogether']:
                train_real = split_dataframe(data, test, sort, "train_real")
                train_fake = split_dataframe(data, test, sort, "train_fake")
                train_data = pd.concat([train_real, train_fake])
            else:
                RuntimeError(f"Experiment ({experiment_type}) type not implemented")

            valid_data  = split_dataframe(data, test, sort, "valid_real")
            test_data   = split_dataframe(data, test, sort, "test_real" )

            # get train state
            train_state = self.get_train_state_from(test,sort)
            train_state = evaluate_tuning(train_state,train_data, valid_data, test_data, task_params)
       
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


