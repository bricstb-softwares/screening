__all__ = []



import os, pickle, json, sys
import tarfile
import typing as T
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.python.keras.metrics as metrics

from dataclasses import dataclass
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.keras.models import model_from_json
from screening.utils import callbacks
from loguru import logger
from pprint import pprint


#
# NOTE: Preprocessing should be:
# take image -> convert to array -> divide by 255 -> rgb to gray -> crop? -> resize
#

#
# dataset preparation
#

def build_dataset(df, 
                  image_shape, 
                  batch_size  : int, 
                  crop_header : bool=False):
    
    def _decode_image(path, label, image_shape, channels : int=3, crop_header :bool=False):
        image_encoded = tf.io.read_file(path)
        image = tf.io.decode_jpeg(image_encoded, channels=channels)
        #image = tf.image.rgb_to_grayscale(image)
        image = tf.cast(image, dtype=tf.float32) / tf.constant(255., dtype=tf.float32)
        if crop_header:
            image =  image.crop((0,0,image.size[0],image.size[1]-70))
        image = tf.image.resize(image, image_shape, method='nearest')
        label = tf.cast(label, tf.int32)
        return image, label
        
    ds_path  = tf.data.Dataset.from_tensor_slices(df["path"])
    ds_label = tf.data.Dataset.from_tensor_slices(df["label"])
    ds       = tf.data.Dataset.zip((ds_path, ds_label))
    ds       = ds.map(lambda p, l: _decode_image(p, l, image_shape, crop_header=crop_header))
    ds       = ds.batch(batch_size, drop_remainder=False)
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



def build_altogether_dataset(df, image_shape, batch_size, crop_header : bool=False):
    def _decode_weighted_image(path, label, weight, image_shape, 
                               channels : int=3, crop_header :bool=False):
        image_encoded = tf.io.read_file(path)
        image = tf.io.decode_jpeg(image_encoded, channels=channels)
        #image = tf.image.rgb_to_grayscale(image)
        image = tf.cast(image, dtype=tf.float32) / tf.constant(255., dtype=tf.float32)
        if crop_header:
            image =  image.crop((0,0,image.size[0],image.size[1]-70))
        image = tf.image.resize(image, image_shape, method='nearest')
        label = tf.cast(label, tf.int32)
        weight = tf.cast(weight, tf.float32)
        return image, label, weight

    ds = tf.data.Dataset.from_tensor_slices((df["path"], df["label"], df["weights"]))
    ds = ds.map(lambda p, l, w: _decode_weighted_image(p, l, w, image_shape))
    ds = ds.batch(batch_size, drop_remainder=False)
    return ds


#
# model preparation
#
def create_cnn(image_shape,version=1):

    logger.info(f"Builing model {version}...")
    model = Sequential()

    # NOTE: first version from old files
    if version==0:
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
    


    elif version==1:
        input_shape = (image_shape[0], image_shape[1], 3)
        model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))

        model.add(layers.Conv2D(64, (3, 3), activation="relu"))
        model.add(layers.MaxPooling2D((2, 2)))
        #model.add(BatchNormalization())
        model.add(layers.Dropout(0.25))

        model.add(layers.Flatten())
        model.add(
            layers.Dense(32, activation="relu", kernel_regularizer="l2", bias_regularizer="l2")
        )
        #model.add(Dropout(0.5))
        model.add(layers.Dense(1, activation="sigmoid"))
    else:
        raise RuntimeError(f"model version {version} not supported.")

    model.summary()
    return model


@dataclass
class ConvNetState:
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
                    version         : int=1,  # version one should be default for the phase one project
                    name            : str='convnets',
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
            '__version__' : version,
            '__name__'    : name,
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

        model_version = model_params.get("model_version",0)
        model        = create_cnn(image_shape, version=model_version)
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

def build_model_from_job( job_path, name : str='convnets'):

    with open( job_path, 'r') as f:
        if f["__name__"] == name:
            version = f["__version__"]
            if version == 1:
                sequence = f['model']['sequence']
                weights  = f['model']['weights']
                history  = f['history']
                params   = f['params']
                # build model
                model = model_from_json( json.dumps(sequence, separators=(',', ':')) )
                model.set_weights(weights)
                return model, history, params
            else:
                raise RuntimeError(f"version {version} not supported.")
        else:
            raise RuntimeError(f"job file name as {f['__name__']} not supported.")



#
# training
#


def train_neural_net(df_train, df_valid, params, basepath : str=os.getcwd()):

    if "image_shape" not in list(params.keys()):
        params["image_shape"] = [params["image_width"], params["image_height"]]

    ds_train = build_dataset(df_train, params["image_shape"], params["batch_size"])
    ds_valid = build_dataset(df_valid, params["image_shape"], params["batch_size"])

    tf.keras.backend.clear_session()

    initial_epoch = 0
    checkpoint_filepath=basepath+'/checkpoint.json'

    if os.path.exists(checkpoint_filepath):
        logger.info("starting from the last checkpoint...")
        model = callbacks.load_model_from_checkpoint(checkpoint_path=basepath)
    else:
        optimizer = adam_v2.Adam(params["learning_rate"])
        tf_metrics = [
            metrics.BinaryAccuracy(name="acc"),
            metrics.AUC(name="auc"),
            metrics.Precision(name="precision"),
            metrics.Recall(name="recall"),
        ]
        model = create_cnn(params["image_shape"], params['model_version'])
        model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=tf_metrics)

    model_checkpoint  =callbacks.EarlyStopping(
                                 monitor='val_loss', 
                                 patience=params['patience'],
                                 mode='min',
                                 checkpoint_path=basepath,
                                 do_checkpoint=True,
                                 )
  
    history = model.fit_generator(
        ds_train,
        epochs=2,#params["epochs"],
        initial_epoch=model_checkpoint.initial_epoch,
        validation_data=ds_valid,
        callbacks=[
            callbacks.MinimumEpochs(params["min_epochs"]),
            model_checkpoint
        ],
        verbose=1,
    ).history

    history["best_epoch"]    = model_checkpoint.best_epoch

    pprint(history)
    train_state = prepare_model(
        model=model,
        history=history,
        params=params,
    )

    return train_state


def train_interleaved(df_train_real, df_train_fake, df_valid_real, params, basepath :str=os.getcwd()):

    if "image_shape" not in list(params.keys()):
        params["image_shape"] = [params["image_width"], params["image_height"]]

    ds_train = build_interleaved_dataset(
        df_train_real, df_train_fake, params["image_shape"], params["batch_size"]
    )
    ds_valid = build_dataset(df_valid_real, params["image_shape"], params["batch_size"])

    tf.keras.backend.clear_session()

    initial_epoch = 0
    if os.path.exists(basepath+'/checkpoint.json'):
        model, initial_epoch = get_checkpoint_from(basepath)
    else:
        optimizer = adam_v2.Adam(params["learning_rate"])
        tf_metrics = [
            metrics.BinaryAccuracy(name="acc"),
            metrics.AUC(name="auc"),
            metrics.Precision(name="precision"),
            metrics.Recall(name="recall"),
        ]

        model = create_cnn(params["image_shape"], params['model_version'])
        model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=tf_metrics)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=params["patience"],
            verbose=2,
            restore_best_weights=True,
        ),
        MinimumEpochs(params["min_epochs"]),
        Checkpoint(basepath, each_epoch=5),
    ]


    history = model.fit(
        ds_train,
        epochs=params["epochs"],
        initial_epoch=initial_epoch,
        validation_data=ds_valid,
        callbacks=callbacks,
        verbose=2,
    )
    history.history["best_epoch"]    = callbacks[0].best_epoch
    history.history["stopped_epoch"] = callbacks[0].stopped_epoch

    train_state = prepare_model(
        model=model,
        history=history.history,
        params=params,
    )

    return train_state


def train_altogether(df_train_real, df_train_fake, df_valid_real, weights, params, basepath:str=os.getcwd()):

    if "image_shape" not in list(params.keys()):
        params["image_shape"] = [params["image_width"], params["image_height"]]

    df_train = pd.concat([df_train_real, df_train_fake], axis=0, ignore_index=True)
    df_train["weights"] = weights
    ds_train = build_altogether_dataset(
        df_train, params["image_shape"], params["batch_size"]
    )
    ds_valid = build_dataset(df_valid_real, params["image_shape"], params["batch_size"])

    tf.keras.backend.clear_session()
    initial_epoch = 0
    if os.path.exists(basepath+'/checkpoint.json'):
        model, initial_epoch = get_checkpoint_from(basepath)
    else:
        optimizer = adam_v2.Adam(params["learning_rate"])
        tf_metrics = [
            metrics.BinaryAccuracy(name="acc"),
            metrics.AUC(name="auc"),
            metrics.Precision(name="precision"),
            metrics.Recall(name="recall"),
        ]

        model = create_cnn(params["image_shape"], params['model_version'])
        model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=tf_metrics)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=params["patience"],
            verbose=2,
            restore_best_weights=True,
        ),
        MinimumEpochs(params["min_epochs"]),
        Checkpoint(basepath, each_epoch=5),
    ]


    history = model.fit(
        ds_train,
        epochs=params["epochs"],
        initial_epoch=initial_epoch,
        validation_data=ds_valid,
        callbacks=callbacks,
        verbose=2,
    )

    history.history["best_epoch"] = callbacks[0].best_epoch
    history.history["stopped_epoch"] = callbacks[0].stopped_epoch

    train_state = prepare_model(
        model=model,
        history=history.history,
        params=params,
    )

    return train_state


def train_fine_tuning(df_train, df_valid, params, model, basepath:str=os.getcwd()):

    if "image_shape" not in list(params.keys()):
        params["image_shape"] = [params["image_width"], params["image_height"]]

    ds_train = build_dataset(df_train, params["image_shape"], params["batch_size"])
    ds_valid = build_dataset(df_valid, params["image_shape"], params["batch_size"])

    tf.keras.backend.clear_session()

    initial_epoch = 0
    if os.path.exists(basepath+'/checkpoint.json'):
        model, initial_epoch = get_checkpoint_from(basepath)
    else:
        optimizer = adam_v2.Adam(params["learning_rate"])
        tf_metrics = [
            metrics.BinaryAccuracy(name="acc"),
            metrics.AUC(name="auc"),
            metrics.Precision(name="precision"),
            metrics.Recall(name="recall"),
        ]

        # lock layers for finetuning
        for layer in model.layers[:-2]:
            layer.trainable = False
        model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=tf_metrics)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=params["patience"],
            verbose=2,
            restore_best_weights=True,
        ),
        MinimumEpochs(params["min_epochs"]),
        Checkpoint(basepath, each_epoch=5),
    ]


    history = model.fit(
        ds_train,
        epochs=params["epochs"],
        initial_epoch=initial_epoch,
        validation_data=ds_valid,
        callbacks=callbacks,
        verbose=2,
    )
    history.history["best_epoch"] = callbacks[0].best_epoch
    history.history["stopped_epoch"] = callbacks[0].stopped_epoch

    train_state = prepare_model(
        model=model,
        history=history.history,
        params=params,
    )

    return train_state


#oss: 0.2204 - acc: 0.9819 - auc: 0.9982 - precision: 0.9777 - recall: 0.9777




