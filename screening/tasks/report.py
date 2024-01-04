
__all__ = []

import os, pickle, tarfile, luigi
import numpy as np
import pandas as pd

from pathlib import Path
from loguru import logger
from screening import TARGET_DIR, Task, CrossValidation
from screening.utils.report import (
    build_cross_val_metrics,
    calculate_performance_metrics,
    plot_roc_curve,
    report_training,
)


class ReportCNN(Task):
    experiment_hash = luigi.Parameter()
    dataset_info = luigi.DictParameter()

    def output(self):
        # type: () -> luigi.LocalTarget
        output_path = Path(self.get_output_path())
        output_name = output_path.name + ".tar.gz"
        output_file = output_path.parent / output_name
        return luigi.LocalTarget(str(output_file))

    def requires(self):
        # type: () -> list[CrossValidation]
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

    def run(self):
        # type: () -> None
        self.set_logger()
        logger.info(f"=== Start '{self.__class__.__name__}' === \n")

        task_params = self.__dict__["param_kwargs"].copy()
        logger.info(f"Experiment Hash: {task_params['experiment_hash']}")
        logger.info("Dataset Info:")
        for dataset in self.dataset_info:
            logger.info(f" {dataset}")
            database = self.dataset_info[dataset]
            tag = database["tag"]
            logger.info(f" - tag: {tag}")
            for source in database["sources"]:
                logger.info(f" - source: {source}")
        logger.info("\n")

        metadata_list = []
        for i in range(len(self.requires())):
            metadata_path = self.requires()[i].output().path
            metadata_list.append(pd.read_parquet(metadata_path))

        metadata = pd.concat(metadata_list)

        # reporting loop
        folds = list(range(10))
        inner_folds = list(range(9))
        experiment_path = Path(TARGET_DIR) / self.experiment_hash
        cv_metrics_val = dict()

        for i in folds:
            sp_who, best_sort_val = -np.inf, None

            for j in inner_folds:
                logger.info(f"Fold {i} / Inner Fold {j}")
                results_path = experiment_path / f"cnn_fold{i}/sort{j}"
                output_path = Path(self.get_output_path()) / f"cnn_fold{i}/sort{j}"
                if not output_path.exists():
                    output_path.mkdir(parents=True)

                # training history
                report_training(results_path, "train_history")

                # validation performance
                data_val = metadata[
                    (metadata["fold"] == i)
                    & (metadata["inner_fold"] == j)
                    & (metadata["set"] == "val")
                ]
                valid_metrics = calculate_performance_metrics(data_val, results_path)
                output_file = output_path / "valid_metrics.pkl"
                with open(output_file, "wb") as file:
                    pickle.dump(valid_metrics, file, protocol=pickle.HIGHEST_PROTOCOL)

                # select CNN by highest sensitivity
                if valid_metrics["SP_WHO"] > sp_who:
                    sp_who = valid_metrics["SP_WHO"]
                    best_sort_val = j
                    cv_metrics_val[f"fold{i}"] = valid_metrics
                    cv_metrics_val[f"fold{i}"]["sort"] = best_sort_val

        # cross validation metrics
        val_results = build_cross_val_metrics(cv_metrics_val)
        output_file = Path(self.get_output_path()) / "valid_results.csv"
        val_results.to_csv(output_file, index=False)

        # mean ROC curve between folds
        roc_interpolated = []
        for fold in folds:
            best_sort = val_results.loc[fold, "sort"]
            aux_path = Path(self.get_output_path()) / f"cnn_fold{fold}/sort{best_sort}"
            fold_metrics = pd.read_pickle(aux_path / "valid_metrics.pkl")
            mean_fpr = np.linspace(0, 1, 100)
            interp_tpr = np.interp(mean_fpr, fold_metrics["fpr"], fold_metrics["tpr"])
            interp_tpr[0] = 0.0
            roc_interpolated.append(interp_tpr)

        mean_tpr = np.mean(roc_interpolated, axis=0)
        mean_tpr[-1] = 1.0
        std_tpr = np.std(roc_interpolated, axis=0)
        upper_error = mean_tpr + std_tpr
        lower_error = mean_tpr - std_tpr

        output_file = Path(self.get_output_path()) / "valid_roc_curve.png"
        plot_roc_curve(mean_tpr, upper_error, lower_error, output_file)

        # save the TAR file output
        output_path = str(self.get_output_path())
        with tarfile.open(self.output().path, "w:gz") as tar:
            for fn in os.listdir(output_path):
                p = os.path.join(output_path, fn)
                tar.add(p, arcname=fn)

        logger.info("")
        logger.info(f" - Output: {self.get_output_path()}")
        logger.info("")
        logger.info(f"=== End: '{self.__class__.__name__}' ===")
        logger.info("")
