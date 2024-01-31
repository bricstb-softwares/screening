


__all__ = [
    # dataset build
    "build_dataset",
    "build_altogether_dataset",
    "build_interleaved_dataset",
    # model
    "create_cnn",
    "prepare_model",
    # save
    "save_job_state",
    "save_train_state",
    # loaders
    "build_model_from_job",
    "build_model_from_train_state",
]



import os, pickle, json, datetime
import typing as T
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf

from dataclasses import dataclass
from pathlib import Path
from loguru import logger
from tensorflow.python.keras import layers
from tensorflow.python.keras.metrics import AUC, BinaryAccuracy, Precision, Recall
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.keras.models import model_from_json



#
# dataset preparation
#



def build_dataset(df, image_shape, batch_size):
    # type: (pd.DataFrame, list[int], int) -> tf.data.Dataset
    def _decode_image(path, label, image_shape, channels=3):
        # type: (str, int, list[int], int) -> T.Union[list, int]
        image_encoded = tf.io.read_file(path)
        image = tf.io.decode_jpeg(image_encoded, channels=channels)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, image_shape)
        label = tf.cast(label, tf.int32)
        return image, label

    ds_path = tf.data.Dataset.from_tensor_slices(df["path"])
    ds_label = tf.data.Dataset.from_tensor_slices(df["label"])
    ds = tf.data.Dataset.zip((ds_path, ds_label))
    ds = ds.map(lambda p, l: _decode_image(p, l, image_shape))
    ds = ds.batch(batch_size, drop_remainder=False)
    return ds



#
# model preparation
#






@dataclass
class SVMState:
    model_sequence  : dict 
    model_weights   : list
    history         : dict
    parameters      : dict
    

def prepare_model(model, history, params):
    # type: (tf.python.keras.models.Sequential, dict, dict) -> ConvNetState
    train_state = ConvNetState(json.loads(model.to_json()), model.get_weights(), history, params)
    return train_state


#
# save
#


# as task 
def save_train_state(output_path : str, train_state : str, tar : bool=False):
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

    if tar:
        tar_path = output_path / f"{output_path.name}.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            for filepath in filepaths:
                tar.add(filepath)


# as job
def save_job_state( path            : str, 
                    train_state     : ConvNetState, 
                    test            : int,
                    sort            : int,
                    metadata        : dict={},
                    version         : int=1  # version one should be default for the phase one project
                ):
    
    # NOTE: version will be used to configure the pre-processing function during the load inference
    metadata.update({'test':test, 'sort':sort})
    d = {
            'model': {
                'weights'   : train_state.model_weights, 
                'sequence'  : train_state.model_sequence,
            },
            'history'     : train_state.history,
            'metadata'    : metadata,
            '__version__' : version
        }

    with open(path, 'wb') as file:
        pickle.dump(d, file, pickle.HIGHEST_PROTOCOL)



#
# Loaders
#



def build_model_from_train_state( train_state ):

    if type(train_state) == str:
        # load weights  
        model_path = train_state
        weights_path = os.path.join(model_path, "model_weights.pkl")
        weights      = pickle.load( open(weights_path, 'rb') )
        # load params
        params_path  = os.path.join(model_path, "parameters.pkl")
        model_params = pickle.load( open(params_path, 'rb' ))
        # build model
        image_shape  = model_params["image_shape"]
        model        = create_cnn(image_shape)
        model.set_weights(weights)
        # load history
        history_path = os.path.join(model_path, 'history.pkl')
        history      = pickle.load(open(history_path, 'rb'))
        return model, history, model_params

    else:
        # get the model sequence
        sequence = train_state.model_sequence
        weights  = train_state.model_weights
        model = model_from_json( json.dumps(sequence, separators=(',', ':')) )
        # load the weights
        model.set_weights(weights)
        return model, train_state.history, train_state.parameters

def build_model_from_job( job_path ):

    with open( job_path, 'r') as f:
        sequence = f['model']['sequence']
        weights  = f['model']['weights']
        history  = f['history']
        params   = f['params']
        # build model
        model = model_from_json( json.dumps(sequence, separators=(',', ':')) )
        model.set_weights(weights)
        return model, history, params



#
# training
#


def train_svm(df_train, df_valid, params):

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
        epochs=2,#params["epochs"],
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





