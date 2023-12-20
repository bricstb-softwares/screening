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
    evaluate_tuning,
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
        
        logging.info(f"Experiment hash: {self.get_hash()}")
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


