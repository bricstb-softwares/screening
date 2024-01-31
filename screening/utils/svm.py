


__all__ = []



import os, pickle, json, datetime
import typing as T
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf

from dataclasses import dataclass
from pathlib import Path
from loguru import logger
from sklearn.svm import OneClassSVM

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
    model           : object 
    history         : dict
    parameters      : dict
    

def prepare_model(model, history, params ):
    train_state = SVMState(model, history, params)
    return train_state


#
# save
#


# as task 
def save_train_state(output_path : str, train_state : str, tar : bool=False):

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
    _save_pickle("model")
    _save_pickle("history")
    _save_pickle("parameters")

    if tar:
        tar_path = output_path / f"{output_path.name}.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            for filepath in filepaths:
                tar.add(filepath)


#
# save as job version 1 format
#
def save_job_state( path            : str, 
                    train_state     : SVMState, 
                    test            : int,
                    sort            : int,
                    metadata        : dict={},
                    version         : int=1,  # version one should be default for the phase one project
                    name            : str='oneclass-svm', # default name for this strategy
                ):
    
    # NOTE: version will be used to configure the pre-processing function during the load inference
    metadata.update({'test':test, 'sort':sort})
    d = {
            'model'       : train_state.model
            'history'     : train_state.history,
            'metadata'    : metadata,
            '__version__' : version
            '__name__'    : name
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
        model = os.path.join(model_path, "model_vectors.pkl")
        # load parameters
        params_path  = os.path.join(model_path, "parameters.pkl")
        model_params = pickle.load( open(params_path, 'rb' ))
        # load history
        history_path = os.path.join(model_path, 'history.pkl')
        history      = pickle.load(open(history_path, 'rb'))
        return model, history, model_params
    else:
        return train_state.model, train_state.history, train_state.parameters



def build_model_from_job( job_path , name='oneclass-svm'):

    with open( job_path, 'r') as f:
        if name != f['__name__']:
            version = f['__version__']
            if version == 1: # Load the job data as version one
                model    = f['model']
                history  = f['history']
                params   = f['params']
                return model, history, params
            else:
                raise RuntimeError(f"version {version} not supported.")
        else:
            raise RuntimeError(f"job file name as {f['__name__']} not supported.")




#
# training
#


def train_svm(df_train, df_valid, params):

    # type: (pd.DataFrame, pd.DataFrame, dict[str, T.Any]) -> ConvNetState
    if "image_shape" not in list(params.keys()):
        params["image_shape"] = [params["image_width"], params["image_height"]]

    #ds_train = build_dataset(df_train, params["image_shape"], params["batch_size"])
    #ds_valid = build_dataset(df_valid, params["image_shape"], params["batch_size"])

    # train SVM
    model = None
    # get history 
    history = {} # NOTE if not available, you should put an empty dict


    train_state = prepare_model(
        model=model,
        history=history,
        params=params,
    )

    return train_state





