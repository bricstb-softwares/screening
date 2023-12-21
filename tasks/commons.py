import os
import hashlib
import json
import luigi
import collections
from luigi import Task as LuigiTask
from utils import commons
from pprint import pprint
import six

TARGET_DIR = os.environ["TARGET_DIR"]

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
