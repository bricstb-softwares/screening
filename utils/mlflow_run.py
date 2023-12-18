import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import mlflow
import pickle
import pandas as pd
import json
import dorothy_sdk.mlflow as dsml
from dorothy_sdk.git_repo import Repository
import dorothy_sdk.dataset_wrapper as dw
from .convnets import create_cnn
from lpsds.logger import log

import tensorflow as tf
import warnings

import absl.logging

absl.logging.set_verbosity(absl.logging.ERROR)

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")


class MLFlow:
    def __init__(self, model_path):
        self.work_dir = "/app"
        self.num_partitions = 10
        self.num_splits = 9
        self.image_shape = (256, 256)
        self.model_code_paths = [self.work_dir + "/screening/utils"]
        self.model_path = model_path
        self.mlf_obj = dsml.MLFlow(flavor=mlflow.tensorflow)

    def read_pickle(self, fname):
        with open(fname, "rb") as f:
            return pickle.load(f)

    def load_model(self, partition, split):
        weights = self.read_pickle(
            os.path.join(
                self.model_path,
                f"cnn_fold{partition}",
                f"sort{split}",
                "model_weights.pkl",
            )
        )
        model = create_cnn(self.image_shape)
        model.set_weights(weights)
        return model

    def save_version_info(self):
        model_rep = Repository(self.work_dir)
        mlflow.log_params(
            {
                "model_repository": model_rep.get_location(),
                "model_version": model_rep.get_version(),
            }
        )

    def save_dataset_info(self):
        mlflow.log_params(
            {
                "train_dataset": json.dumps(["china", "osp_shenzen_pix2pix_v1"]),
                "validation_dataset": json.dumps(["china"]),
                "test_dataset": json.dumps(["china"]),
            }
        )

    def save_hyperparams(self):
        hyper_param_file_name = os.path.join(
            self.model_path, "cnn_fold0/sort0", "parameters.pkl"
        )
        hyper_params = self.read_pickle(hyper_param_file_name)
        mlflow.log_params(hyper_params)
        mlflow.log_param("device_used", "cuda")

    def save_models(self):
        for partition in range(self.num_partitions):
            for split in range(self.num_splits):
                model = self.load_model(partition, split)
                self.mlf_obj.log_model(
                    model, partition, split, code_paths=self.model_code_paths
                )

    def save_metrics(self):
        metrics = pd.read_csv(os.path.join(self.model_path, "test_results.csv"))
        cv_metrics = {}
        for c in metrics.columns:
            cv_metrics["test_" + c.lower()] = metrics[c].to_numpy()
        self.mlf_obj.log_statistics(cv_metrics)
        return metrics

    def save_operation_model(self, metrics):
        best_model_partition = metrics.SP.argmax()
        best_model_split = metrics.loc[metrics.SP == metrics.SP.max(), "sort"].iloc[0]
        op_metrics_fname = os.path.join(
            self.model_path,
            f"cnn_fold{best_model_partition}/sort{best_model_split}/metrics",
            "test_metrics.pkl",
        )
        threshold = self.read_pickle(op_metrics_fname)["threshold"]

        mlflow.log_params(
            {
                "best_fold_id": best_model_partition,
                "best_split_id": best_model_split,
                "decision_threshold": threshold,
                "flavor": "tensorflow",
            }
        )
        self.mlf_obj.log_model(
            self.load_model(best_model_partition, best_model_split),
            code_paths=self.model_code_paths,
        )

    def save_models_outputs(self, metrics):
        part_list = []
        for partition, row in metrics.iterrows():
            split = int(row.sort)
            fname = os.path.join(
                self.model_path,
                f"cnn_fold{partition}/sort{split}/metrics",
                "test_metrics.pkl",
            )
            pred_df = self.read_pickle(fname)["predictions"]
            pred_df["fold"] = partition
            part_list.append(pred_df)

        df = pd.concat(part_list, ignore_index=True)
        df.rename(columns={"image_name": "image", "y_prob": "y_proba"}, inplace=True)
        df["dataset"] = df.image.str.extract(r"(\w+?)_.+")

        ds_list = []
        for dataset in df.dataset.unique():
            ds_list.append(dw.Dataset(dataset).load_metadata())
        datasets_df = pd.concat(ds_list, ignore_index=True)[["image", "y_true"]]

        df = df.merge(datasets_df, how="inner", on="image")
        df["y_pred"] = df.y_pred.map({False: 0, True: 1})
        self.mlf_obj.log_dataframe(df, "fold_model_output")
        return df


def log_run(model_path):
    exp_name = "PhilCNN"
    log.info('Iniciando o registro do treinamento salvo em "%s".', model_path)
    log.info('    Inicializando um novo run para o experimento "%s".', exp_name)
    mlflow.set_experiment(exp_name)
    with mlflow.start_run():
        run = MLFlow(model_path)

        log.info("    Salvando as informações referentes ao modelo e sua versão.")
        run.save_version_info()

        log.info(
            "    Salvando as informações a respeito dos datasets empregados no desenvolvimento."
        )
        run.save_dataset_info()

        log.info("    Salvando hiperparâmetros do treinamento.")
        run.save_hyperparams()

        log.info(
            "    Salvando todas os modelos produzidos (%s).",
            run.num_partitions * run.num_splits,
        )
        run.save_models()

        log.info(
            "    Salvando as figuras de mérito de cada uma das %s partições",
            run.num_partitions,
        )
        metrics = run.save_metrics()

        log.info("    Salvando as informações a respeito do modelo de operação.")
        run.save_operation_model(metrics)

        log.info(
            "    Salvando a saída de cada modelo de partição para o seu respectivo conjunto de teste."
        )
        run.save_models_outputs(metrics)

        log.info("Fim do registro do treinamento no MLFlow.")
