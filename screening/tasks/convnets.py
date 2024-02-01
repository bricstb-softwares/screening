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
from screening.utils.convnets import (
    save_train_state,
    save_job_state,
    build_model_from_job,
    build_model_from_train_state,
    train_altogether,
    train_fine_tuning,
    train_interleaved,
    train_neural_net,
)

from screening.utils.data import (
    split_dataframe,
)
from screening.validation.evaluate import (
    evaluate
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

        train_state = train_neural_net(train_fake, valid_real, task_params)
        train_state = evaluate( train_state, train_fake, valid_real, test_real)

        end = default_timer()
        logger.info(f"training toke {timedelta(seconds=(end - start))}...")

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
        train_state = evaluate( train_state, train_real_fake, valid_real, test_real)

        end = default_timer()
        logger.info(f"training toke {timedelta(seconds=(end - start))}...")

        return train_state
        
class TrainAltogether(TrainCNN):


    def fit(self, data, test, sort, task_params ):

        start = default_timer()

        train_real = split_dataframe(data, test, sort, "train_real")
        train_fake = split_dataframe(data, test, sort, "train_fake")
        valid_real = split_dataframe(data, test, sort, "valid_real")
        test_real  = split_dataframe(data, test, sort, "test_real" )
        real_weights = (1 / len(train_real)) * np.ones(len(train_real))
        fake_weights = (1 / len(train_fake)) * np.ones(len(train_fake))
        weights = np.concatenate((real_weights, fake_weights))
        weights = weights / sum(weights)
        train_state = train_altogether(
            train_real, train_fake, valid_real, weights, task_params
        )
        train_real_fake = pd.concat([train_real, train_fake])
        train_state = evaluate( train_state, train_real_fake, valid_real, test_real)

        end = default_timer()
        logger.info(f"training toke {timedelta(seconds=(end - start))}...")

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
        train_state = evaluate( train_state, train_fake, valid_real, test_real)

        end = default_timer()
        logger.info(f"training toke {timedelta(seconds=(end - start))}...")

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

        train_real = split_dataframe(data, test, sort, "train_real")
        valid_real = split_dataframe(data, test, sort, "valid_real")
        test_real  = split_dataframe(data, test, sort, "test_real" )
        model      = self.get_parent_model(test, sort)
                    
        train_state = train_fine_tuning(
                    train_real, valid_real, task_params, model
                )
        train_state = evaluate( train_state, train_real, valid_real, test_real)

        end = default_timer()
        logger.info(f"training toke {timedelta(seconds=(end - start))}...")

        return train_state


