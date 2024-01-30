

__all__ = ["inference_tab"]

import gradio as gr
import io
import sys, os, pickle, json
import tensorflow as tf
import numpy as np

from loguru  import logger
from config import server_flags as flags
from config import update_log



model_from_json = tf.keras.models.model_from_json
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices)>0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.run_functions_eagerly(False)


def load_model( path ):
    
    
    def preproc_input( path ,channels=3, image_shape=(256,256)):
        image_encoded = tf.io.read_file(path)
        image = tf.io.decode_jpeg(image_encoded, channels=channels)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, image_shape)
        image = np.expand_dims(image, axis=0)

        return image

    
    with open(path, 'rb') as f:
        d = pickle.load(f)
        metadata = d['metadata']
        model = d["model"]
        logger.info("creating model...")
        model = model_from_json( json.dumps( d['model']['sequence'], separators=(',',':')) )
        model.set_weights( d['model']['weights'] )

        # Get the threshold
        

        model.summary()
        return model, preproc_input





    
def create_inference_vars():

    model, preproc = load_model( flags.model )

    ctx = {
       "model"   : model,
       "preproc" : preproc,
    }

    return ctx


def predict( context, image):

    logger.info("predicting....")

    model = context["model"]
    preproc = context["preproc"]

    # NOTE: predict
    prob = model.predict( preproc(image) )[0][0] # NOTE: get single value since this is a one sample mode

    logger.info(f"image {image} output prob as {prob}")

    response = f"Model given a prob of {prob}"

    return context, response, update_log()


#
# NOTE: Build labelling tab here with all trigger functions inside
#
def inference_tab(context ,  name = 'Inference'):


    with gr.Tab(name):
        with gr.Row():
            with gr.Column():
                image       = gr.Image(label='', type='filepath')
                predict_btn = gr.Button("Predict")
            with gr.Column():
                response     = gr.Textbox(label='Source')

        with gr.Row():
            log = gr.Textbox(label='data logger', max_lines=10)

        predict_btn.click( predict , [context, image], [context, response, log])
     
