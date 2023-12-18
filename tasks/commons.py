import os
import hashlib
import json
import luigi
from luigi import Task as LuigiTask
from utils import commons

TARGET_DIR = os.environ["TARGET_DIR"]

class Task(LuigiTask):
    


    def get_output_path(self):    
        hash_experiment = self.get_task_family() + "_%s" % self.get_hash()
        output_path = os.path.join(TARGET_DIR, hash_experiment) if not self.get_job_params() else os.getcwd()
        return output_path


    def get_job_params(self):
        job_params = self.__dict__["param_kwargs"].copy()['job']
        return job_params


    def get_hash(self):
        params = self.to_str_params(only_significant=True)
        required_tasks = self.requires()
        if required_tasks:
            for i in self.requires():
                required_params = i.to_str_params(only_significant=True)
                params.update(required_params)
        return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:10]


    def set_logger(self):
        commons.create_folder(self.get_output_path())
        commons.set_task_logger(
            log_prefix=f"{self.__class__.__name__}", log_path=self.get_output_path()
        )
