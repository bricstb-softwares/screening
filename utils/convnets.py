from __future__ import annotations

import os
import pickle
import json
import typing as T
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from joblib import Memory
from tensorflow.python.keras import layers
from tensorflow.python.keras.metrics import AUC, BinaryAccuracy, Precision, Recall
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.keras.models import model_from_json

from utils import report
from pprint import pprint

MEMORY = Memory("/tmp/LTBI", verbose=0)


def split_dataframe(df, fold, inner_fold, type_set):
    # type: (pd.DataFrame, int, int, str) -> pd.DataFrame
    if type_set == "train_real":
        res = df[
            (df["type"] == "real")
            & (df["set"] == "train")
            & (df["fold"] == fold)
            & (df["inner_fold"] == inner_fold)
        ]
    elif type_set == "valid_real":
        res = df[
            (df["type"] == "real")
            & (df["set"] == "val")
            & (df["fold"] == fold)
            & (df["inner_fold"] == inner_fold)
        ]
    elif type_set == "train_fake":
        res = df[
            (df["type"] == "fake")
            & (df["set"] == "train")
            & (df["fold"] == fold)
            & (df["inner_fold"] == inner_fold)
        ]
    else:
        raise NotImplementedError(f"Type set '{type_set}' not implemented.")

    return res[["source", "path", "label"]]


def filter_metadata(metadata, partition, fold, sort):
    # type: (pd.DataFrame, str, int, int) -> pd.DataFrame
    metadata = metadata[
        (metadata["dataset"] == partition.lower())
        & (metadata["test"] == fold)
        & (metadata["sort"] == sort)
        & (metadata["type"] == "real")
    ]
    metadata = metadata.rename(columns={"raw_image_path": "path", "target": "label"})

    return metadata


def _decode_image(path, label, image_shape, channels=3):
    # type: (str, int, list[int], int) -> T.Union[list, int]
    image_encoded = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image_encoded, channels=channels)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, image_shape)
    label = tf.cast(label, tf.int32)

    return image, label


def _decode_weighted_image(path, label, weight, image_shape, channels=3):
    # type: (str, int, float, list[int], int) -> T.Union[list, int, float]
    image_encoded = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image_encoded, channels=channels)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, image_shape)
    label = tf.cast(label, tf.int32)
    weight = tf.cast(weight, tf.float32)

    return image, label, weight


def build_dataset(df, image_shape, batch_size):
    # type: (pd.DataFrame, list[int], int) -> tf.data.Dataset
    ds_path = tf.data.Dataset.from_tensor_slices(df["path"])
    ds_label = tf.data.Dataset.from_tensor_slices(df["label"])
    ds = tf.data.Dataset.zip((ds_path, ds_label))
    ds = ds.map(lambda p, l: _decode_image(p, l, image_shape))
    ds = ds.batch(batch_size, drop_remainder=False)

    return ds


def build_interleaved_dataset(df_real, df_fake, image_shape, batch_size):
    # type: (pd.DataFrame, pd.DataFrame, list[int], int) -> tf.data.Dataset
    ds_real = build_dataset(df_real, image_shape, batch_size)

    sources = df_fake["source"].unique()
    if len(sources) == 1:
        ds_fake = build_dataset(df_fake, image_shape, batch_size)
        datasets = [ds_real, ds_fake]
    else:
        datasets = [
            build_dataset(df_fake[df_fake["source"] == s], image_shape, batch_size)
            for s in sources
        ]
        datasets.insert(0, ds_real)

    repeat = np.ceil(df_real.shape[0] / batch_size)
    choice_dataset = tf.data.Dataset.range(len(datasets)).repeat(repeat)
    ds = tf.data.Dataset.choose_from_datasets(datasets, choice_dataset)

    return ds


def build_altogether_dataset(df, image_shape, batch_size):
    # type: (pd.DataFrame, list[int], int) -> tf.data.Dataset
    ds = tf.data.Dataset.from_tensor_slices((df["path"], df["label"], df["weights"]))
    ds = ds.map(lambda p, l, w: _decode_weighted_image(p, l, w, image_shape))
    ds = ds.batch(batch_size, drop_remainder=False)

    return ds


def build_model( sequence, weights ):
    model = model_from_json( json.dumps(sequence, separators=(',', ':')) )
    model.set_weights(weights)
    return model


def create_cnn(image_shape):
    # type: (list[int]) -> tf.python.keras.models.Sequential
    model = Sequential()
    model.add(
        layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation="relu",
            input_shape=(image_shape[0], image_shape[1], 3),
        )
    )
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(
        layers.Conv2D(
            filters=128, kernel_size=(3, 3), activation="relu", padding="same"
        )
    )
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(
        layers.Conv2D(
            filters=128, kernel_size=(3, 3), activation="relu", padding="same"
        )
    )
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(units=128, activation="relu"))
    model.add(layers.Dense(units=1, activation="sigmoid"))

    model.summary()
    return model



def create_fine_tunning_cnn(synthetic_path):
    # type: (str) -> tf.python.keras.models.Sequential
    weights_path = os.path.join(synthetic_path, "model_weights.pkl")
    model_params = pd.read_pickle(os.path.join(synthetic_path, "parameters.pkl"))
    image_shape = model_params["image_shape"]

    model = create_cnn(image_shape)
    model.set_weights(pd.read_pickle(weights_path))

    for layer in model.layers[:-2]:
        layer.trainable = False

    return model


@dataclass
class ConvNetState:
    model_sequence: dict 
    model_weights: list
    history: dict[str, list]
    parameters: dict


def prepare_model(model, history, params):
    # type: (tf.python.keras.models.Sequential, dict, dict) -> ConvNetState
    train_state = ConvNetState(json.loads(model.to_json()), model.get_weights(), history, params)
    return train_state


#@MEMORY.cache
def train_neural_net(df_train, df_valid, params):
    # type: (pd.DataFrame, pd.DataFrame, dict[str, T.Any]) -> ConvNetState
    if "image_shape" not in list(params.keys()):
        params["image_shape"] = [params["image_width"], params["image_height"]]

    ds_train = build_dataset(df_train, params["image_shape"], params["batch_size"])
    ds_valid = build_dataset(df_valid, params["image_shape"], params["batch_size"])

    tf.keras.backend.clear_session()
    optimizer = adam_v2.Adam(params["learning_rate"])
    tf_metrics = [
        BinaryAccuracy(name="acc"),
        AUC(name="auc"),
        Precision(name="precision"),
        Recall(name="recall"),
    ]

    model = create_cnn(params["image_shape"])
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=tf_metrics)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    history = model.fit(
        ds_train,
        epochs= 2,#params["epochs"],
        validation_data=ds_valid,
        callbacks=[early_stop],
        verbose=1,
    )
    history.history["best_epoch"] = early_stop.best_epoch

    train_state = prepare_model(
        model=model,
        history=history.history,
        params=params,
    )

    return train_state


#@MEMORY.cache
def train_interleaved(df_train_real, df_train_fake, df_valid_real, params):
    # type: (pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, T.Any]) -> ConvNetState
    if "image_shape" not in list(params.keys()):
        params["image_shape"] = [params["image_width"], params["image_height"]]

    ds_train = build_interleaved_dataset(
        df_train_real, df_train_fake, params["image_shape"], params["batch_size"]
    )
    ds_valid = build_dataset(df_valid_real, params["image_shape"], params["batch_size"])

    tf.keras.backend.clear_session()
    optimizer = adam_v2.Adam(params["learning_rate"])
    tf_metrics = [
        BinaryAccuracy(name="acc"),
        AUC(name="auc"),
        Precision(name="precision"),
        Recall(name="recall"),
    ]

    model = create_cnn(params["image_shape"])
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=tf_metrics)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    history = model.fit(
        ds_train,
        epochs=params["epochs"],
        validation_data=ds_valid,
        callbacks=[early_stop],
        verbose=2,
    )
    history.history["best_epoch"] = early_stop.best_epoch

    train_state = prepare_model(
        model=model,
        history=history.history,
        params=params,
    )

    return train_state


#@MEMORY.cache
def train_altogether(df_train_real, df_train_fake, df_valid_real, weights, params):
    # type: (pd.DataFrame, pd.DataFrame, pd.DataFrame, list, dict[str, T.Any]) -> ConvNetState
    if "image_shape" not in list(params.keys()):
        params["image_shape"] = [params["image_width"], params["image_height"]]

    df_train = pd.concat([df_train_real, df_train_fake], axis=0, ignore_index=True)
    df_train["weights"] = weights
    ds_train = build_altogether_dataset(
        df_train, params["image_shape"], params["batch_size"]
    )
    ds_valid = build_dataset(df_valid_real, params["image_shape"], params["batch_size"])

    tf.keras.backend.clear_session()
    optimizer = adam_v2.Adam(params["learning_rate"])
    tf_metrics = [
        BinaryAccuracy(name="acc"),
        AUC(name="auc"),
        Precision(name="precision"),
        Recall(name="recall"),
    ]

    model = create_cnn(params["image_shape"])
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=tf_metrics)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    history = model.fit(
        ds_train,
        epochs=params["epochs"],
        validation_data=ds_valid,
        callbacks=[early_stop],
        verbose=2,
    )

    history.history["best_epoch"] = early_stop.best_epoch

    train_state = prepare_model(
        model=model,
        history=history.history,
        params=params,
    )

    return train_state


#@MEMORY.cache
def train_fine_tuning(df_train, df_valid, params, synthetic_path):
    # type: (pd.DataFrame, pd.DataFrame, dict[str, T.Any], str) -> ConvNetState
    if "image_shape" not in list(params.keys()):
        params["image_shape"] = [params["image_width"], params["image_height"]]

    ds_train = build_dataset(df_train, params["image_shape"], params["batch_size"])
    ds_valid = build_dataset(df_valid, params["image_shape"], params["batch_size"])

    tf.keras.backend.clear_session()
    optimizer = adam_v2.Adam(params["learning_rate"])
    tf_metrics = [
        tf.keras.metrics.BinaryAccuracy(name="acc"),
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ]
    model = create_fine_tunning_cnn(synthetic_path)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=tf_metrics)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    history = model.fit(
        ds_train,
        epochs=params["epochs"],
        validation_data=ds_valid,
        callbacks=[early_stop],
        verbose=2,
    )
    history.history["best_epoch"] = early_stop.best_epoch

    train_state = prepare_model(
        model=model,
        history=history.history,
        params=params,
    )

    return train_state


def evaluate_neural_net(df_eval, model_weights, params):
    # type: (pd.DataFrame, list, dict[str, T.Any]) -> dict[str, T.Any]
    ds_eval = build_dataset(
        df=df_eval, image_shape=params["image_shape"], batch_size=params["batch_size"]
    )

    model = create_cnn(params["image_shape"])
    model.set_weights(model_weights)

    metrics = report.calculate_metrics(ds_eval, df_eval, model)

    return metrics


def save_train_state(output_path, train_state):
    # type: (Path, ConvNetState) -> None
    if not output_path.exists():
        output_path.mkdir(parents=True)

    filepaths = []

    def _save_pickle(attribute_name) -> None:
        path = output_path / f"{attribute_name}.pkl"
        with open(path, "wb") as file:
            pickle.dump(
                getattr(train_state, attribute_name), file, pickle.HIGHEST_PROTOCOL
            )
        filepaths.append(path)

    _save_pickle("model_weights")
    _save_pickle("history")
    _save_pickle("parameters")

    # tar_path = output_path / f"{output_path.name}.tar.gz"
    # with tarfile.open(tar_path, "w:gz") as tar:
    #     for filepath in filepaths:
    #         tar.add(filepath)

    return None


def save_job_state( path, dataset , train_state, job_params, task_params, dataset_info, hash_experiment, metadata={}):
    
    test = job_params['test']
    sort = job_params['sort']

    metrics = get_training_metrics( build_model( train_state.model_sequence, train_state.model_weights), 
                                        dataset, task_params, test, sort )
    train_state.history['summary'] = metrics

    with open(path, 'wb') as file:

        d = {
            'weights'   : train_state.model_weights, 
            'sequence'  : train_state.model_sequence,
            'test'      : test,
            'sort'      : sort,
            'history'   : train_state.history,
            'time'      : task_params['training_time'],
            'hash'      : hash_experiment,
            'metadata'  : metadata
        }

        pickle.dump(d, file, pickle.HIGHEST_PROTOCOL)


def get_training_metrics( model , dataset, params,  test, sort):

    # validation performance
    data_train = dataset[
                      (dataset["fold"] == test)
                    & (dataset["inner_fold"] == sort)
                    & (dataset["set"] == "train")
                ]

    data_val = dataset[
                      (dataset["fold"] == test)
                    & (dataset["inner_fold"] == sort)
                    & (dataset["set"] == "val")
                ]

    data_test = dataset[
                      (dataset["fold"] == test)
                    & (dataset["inner_fold"] == sort)
                    & (dataset["set"] == "val")
                ]             

    # train
    ds = build_dataset(data_train, params["image_shape"], batch_size=128)
    metrics = report.calculate_metrics(ds, data_train, model)

    # validation
    ds = build_dataset(data_val, params["image_shape"], batch_size=128)
    metrics.update( report.calculate_metrics(ds, data_val, model , label='_val' ) )

    # test
    ds = build_dataset(data_test, params["image_shape"], batch_size=128)
    metrics.update( report.calculate_metrics(ds, data_test, model , label='_test' ) )

    # operation (train plus val)
    data_operation = pd.concat([data_train, data_val])
    ds = build_dataset(data_operation, params["image_shape"], batch_size=128)
    metrics.update( report.calculate_metrics(ds, data_operation, model , label='_op' ) )
 
    return metrics






def get_metrics_dict() -> dict[str, T.Any]:
    # type: () -> dict[str, T.Any]
    metrics = {
        "auc": [],
        "sp": [],
        "sp_oms": [],
        "auc_oms": [],
        "sensitivity": [],
        "specificity": [],
        "precision": [],
        "recall": [],
        "tprs": [],
    }

    return metrics


def update_metrics_results(results, metrics):
    # type: (dict[str, T.Any], dict[str, T.Any]) -> dict[str, T.Any]
    metrics["auc"].append(results["auc"])
    metrics["sp"].append(results["sp"])
    metrics["auc_oms"].append(results["auc_oms"])
    metrics["sp_oms"].append(results["sp_oms"])
    metrics["sensitivity"].append(results["sensitivity"])
    metrics["specificity"].append(results["specificity"])
    metrics["precision"].append(results["precision"])
    metrics["recall"].append(results["recall"])
    metrics["tprs"].append(results["tprs"])

    return metrics





