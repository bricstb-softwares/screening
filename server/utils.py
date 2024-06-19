

import socket
import tensorflow as tf
import sys, socket, pickle, json
import collections
import numpy as np

from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils import normalize
from vis.utils import utils
from loguru  import logger
from copy import copy



def setup_logs(  level : str='INFO'):
    """Setup and configure the logger"""
    server_name = socket.gethostname()
    logger.configure(extra={"server_name" : server_name})
    logger.remove()  # Remove any old handler
    format="<green>{time}</green> | <level>{level:^12}</level> | <cyan>{extra[server_name]:<30}</cyan> | <blue>{message}</blue>"
    logger.add(
        sys.stdout,
        colorize=True,
        backtrace=True,
        diagnose=True,
        level=level,
        format=format,
    )
    output_file = 'output.log'
    logger.add(output_file, 
               rotation="50 MB", 
               format=format, 
               level=level, 
               colorize=False)



def get_saliency( model, image, threshold=0.1 ):
    img=image
    if len(img.shape) == 3:
        img = img[np.newaxis]
    layer_idx = utils.find_layer_idx(model, model.layers[-1].name)
    model.layers[layer_idx].activation = tf.keras.activations.linear
    model = utils.apply_modifications(model)
    score = CategoricalScore([0])
    saliency = Saliency(model, clone=False)
    saliency_map = saliency(score, image, smooth_samples=20, smooth_noise=0.2)
    saliency_map = normalize(saliency_map)
    return saliency_map[0]

def paint_saliency( image, saliency_map, threshold=0.1):
    image_mod = copy(image)
    for i in range(image_mod.shape[0]):
        for j in range(image_mod.shape[1]):
            if saliency_map[i][j]>(1-threshold):
                image_mod[i][j][0]=1
    return image_mod

def load_model( path ):

    def preproc_for_convnets( path ,channels=3, image_shape=(256,256), crop : bool=False):
        image_encoded = tf.io.read_file(path)
        image = tf.io.decode_jpeg(image_encoded, channels=channels)
        # image = tf.image.rgb_to_grayscale(image)
        image = tf.cast(image, dtype=tf.float32) / tf.constant(255., dtype=tf.float32)
        if crop:
            shape = tf.shape(image) 
            image = tf.image.crop_to_bounding_box(image, 0,0,shape[0]-70,shape[1])
        image = tf.image.resize(image, image_shape, method='nearest')
        return image.numpy()
    

    preproc = {
        'convnets': preproc_for_convnets
    }

    logger.info(f"reading file from {path}...")
    # NOTE: Open the current file and extract all parameters
    with open(path, 'rb') as f:
        d = pickle.load(f)

        name = d["__name__"]
        version = d["__version__"]
        logger.info(f"name : {name}")

        if name == "convnets":
            logger.info(f"strategy is {name}...")
            # NOTE: This is the current version of the convnets strategy
            if version == 1:
                metadata = d['metadata']
                model = d["model"]
                logger.info("creating model...")
                model = tf.keras.models.model_from_json( json.dumps( d['model']['sequence'], separators=(',',':')) )
                model.set_weights( d['model']['weights'] )
                #model.summary()
                sort = metadata['sort']; test = metadata['test']
                logger.info(f"sort = {sort} and test = {test}")

                # NOTE: get all thresholds here
                history = d['history']
                threshold = {}
                logger.info("current operation points:")

                for key, values in history.items():
                    # HACK: this is a hack to extract the operation keys from the history. Should be improve in future.
                    if (type(values)==collections.OrderedDict) and (key!='summary') and (not 'val' in key): # get all thresholds dicts
                        threshold[key]=values['threshold']
                        logger.info(f"   {key} : {values['threshold']}")

                meta = d['metadata']
                tag = f"{name}-{meta['type']}-test{test}-sort{sort}"
                logger.info(f"model tag : {tag}")
                # return values
                return model, preproc[name], threshold, tag
            else:
                logger.error(f"version {version} not supported.")
        elif name == "oneclass-svm":
            logger.error("not implemented yet.")
        else:
            logger.error(f"name {name} not supported.")
