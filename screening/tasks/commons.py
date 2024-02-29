from __future__ import annotations

__all__ = ["Task"]

import os, hashlib, json, six, luigi
import pandas as pd
from luigi           import Task as LuigiTask
from screening       import TARGET_DIR
from screening.utils import commons
from loguru          import logger

class Task(LuigiTask):
    
    def get_output_path(self):    
        hash_experiment = self.get_task_family() + "_%s" % self.get_hash()
        output_path = os.path.join(TARGET_DIR, hash_experiment) if not self.get_job_params() else os.getcwd()
        return output_path


    def get_job_params(self):
        job_params = self.__dict__["param_kwargs"].copy()['job']
        return job_params


    def get_hash(self, only_significant : bool=True, only_public : bool=False):
        params = dict(self.get_params())
        visible_params = {}
        for param_name, param_value in six.iteritems(self.param_kwargs):
            if (((not only_significant) or params[param_name].significant)
                    and ((not only_public) or params[param_name].visibility == luigi.parameter.ParameterVisibility.PUBLIC)
                    and params[param_name].visibility != luigi.parameter.ParameterVisibility.PRIVATE):
                if type(param_value) == luigi.freezing.FrozenOrderedDict:
                    param_value=dict(param_value)
                visible_params[param_name] = param_value
        visible_params = commons.sort_dict(visible_params)
        visible_params = str(visible_params)        
        return hashlib.md5(json.dumps(visible_params, sort_keys=True).encode()).hexdigest()[:10]


    def set_logger(self):
        commons.create_folder(self.get_output_path())
        commons.set_task_logger(
            log_prefix=f"{self.__class__.__name__}", log_path=self.get_output_path()
        )


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
        from screening.tasks.data import CrossValidation
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

    def output(self) -> luigi.LocalTarget:
        file_name = "output.pkl" if self.get_job_params() else "task_params.pkl"
        output_file = Path(self.get_output_path()) / file_name
        return luigi.LocalTarget(str(output_file))
