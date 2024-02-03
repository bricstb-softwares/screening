__all__ = []

import os, pickle, json, datetime
import typing as T
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import multiprocessing

from dataclasses import dataclass
from pathlib import Path
from loguru import logger
from sklearn.svm import SVC
from sklearn.decomposition import PCA, IncrementalPCA

#
# dataset preparation
#

def parallel_read(frame, resize):
    if len(frame) <= 1:
          return None
    
    data, trgt = [], []
    for row_idx, row in frame.iterrows():
        # Reading image
        image = PIL.Image.open(row['path']).convert('L')
        
        # Resizing image (Project decision)
        image = image.resize(resize, PIL.Image.ANTIALIAS)
        
        # Normalization
        image = np.array(image) / 255.
        
        # Appending data in snake order
        data.append(image.flatten())
        trgt.append(row['label'])
    return np.array(data), np.array(trgt)

def build_dataset(frame, image_shape):
    n_threads = multiprocessing.cpu_count()//2
    data, trgt = [], []
    with Pool(processes = n_threads) as pool:
        multiple_results = [
            pool.apply_async(parallel_read, (break_frame, resize,)) for break_frame in np.array_split(frame, n_threads)
        ]
        results = [res.get() for res in multiple_results]
    data = np.concatenate([element[0] for element in results if element], axis=0)
    trgt = np.concatenate([element[1] for element in results if element], axis=0)
    return data, trgt
    
#
# model preparation
#

@dataclass
class SVMState:
    model           : object 
    history         : dict
    parameters      : dict
    

def prepare_model(model, history, params):
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
        output_path = train_state
        model_path = os.path.join(output_path, "model.pkl")
        model = pickle.load(open(model_path, 'rb'))
        # load parameters
        params_path  = os.path.join(output_path, "parameters.pkl")
        model_params = pickle.load( open(params_path, 'rb' ))
        # load history
        history_path = os.path.join(output_path, 'history.pkl')
        history      = pickle.load(open(history_path, 'rb'))
        return model, history, model_params
    else:
        return train_state.model, train_state.history, train_state.parameters

def build_model_from_job( job_path , name='oneclass-svm'):

    with open( job_path, 'r') as f:
        if name == f['__name__']:
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
        
    # Parameters (should be in params)
    energy_level = params.get('energy_level',95)
    kernel = params.get('kernel',rbf)
    gamma = params.get('gamma', 'scale')
    C = params.get('C', 10)
    
    
    ds_train_x, ds_train_y = build_dataset(df_train, params["image_shape"], params["batch_size"])
    ds_valid_x, ds_valid_y = build_dataset(df_valid, params["image_shape"], params["batch_size"])

    # running pca
    pca_model = PCA(svd_solver = "randomized", whiten = True)
    pca_model.fit(ds_train_x)
    # Selecting number of components.
    energy_cum_sum = np.cumsum(pca_model.explained_variance_ratio_)
    selected_point = np.argmin(np.abs(energy_cum_sum - (energy_level / 100)))
    # Transforming data
    ds_train_x = pca_model.transform(ds_train_x)[:,:selected_point]
    ds_valid_x = pca_model.transform(ds_valid_x)[:,:selected_point]
    
    # train SVM
    model = SVC(
        kernel = kernel,
        gamma = gamma, 
        C = C,
        max_iter = 1e5,
    )
    model.fit(ds_train_x, ds_train_y)
    # get history 
    history = {} # NOTE if not available, you should put an empty dict


    train_state = prepare_model(
        model=model,
        history=history,
        params=params,
    )

    return train_state





