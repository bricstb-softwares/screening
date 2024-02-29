


import gradio as gr
import io
import sys, os, pickle, json
import tensorflow as tf
import numpy as np
import collections

from loguru  import logger
from config import server_flags as flags
from config import update_log
from fastapi import FastAPI

logger.add(flags.log_path,  rotation="50 MB", backtrace=True, diagnose=True)



#
# NOTE: configure the GPU card
#
model_from_json = tf.keras.models.model_from_json
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices)>0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.run_functions_eagerly(False)


def load_model( path ):
    def preproc_for_convnets( path ,channels=3, image_shape=(256,256)):
        image_encoded = tf.io.read_file(path)
        image = tf.io.decode_jpeg(image_encoded, channels=channels)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, image_shape)
        image = np.expand_dims(image, axis=0)
        return image
    preproc = {
        'convnet': preproc_for_convnets
    }

    logger.info(f"reading file from {path}...")
    # NOTE: Open the current file and extract all parameters
    with open(path, 'rb') as f:
        d = pickle.load(f)

        name = d["__name__"]
        version = d["__version__"]

        if name == "convnet":
            logger.info(f"strategy is {name}...")
            # NOTE: This is the current version of the convnets strategy
            if version == 1:
                metadata = d['metadata']
                model = d["model"]
                logger.info("creating model...")
                model = model_from_json( json.dumps( d['model']['sequence'], separators=(',',':')) )
                model.set_weights( d['model']['weights'] )
                model.summary()
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


#
# This should be global to all sessions
#
model, preproc, threshold, model_tag = load_model( flags.model )
flavors        = list(threshold.keys())
default_flavor = flavors[0]
    



def predict( context, flavor, image):
    logger.info("predicting....")
    model     = context["model"]
    preproc   = context["preproc"]
    threshold = context["threshold"][flavor]

    if not image:
        logger.info("Not possible to predict. First, upload a new image and repeat the operation...")
        return context, "","",update_log()
    logger.info(f"operation point is {flavor}")

    # NOTE: predict
    score = model.predict( preproc(image) )[0][0] # NOTE: get single value since this is a one sample mode
    logger.info(f"image {image} output prob as {score}")
    context["score"] = score
    recomendation = "Has tuberculoses" if score>threshold else "Not tuberculoses"
    logger.info(f"model recomentation is : {recomendation}")
    return context, str(score), recomendation, update_log()


def change_flavor(context, flavor):
    score = context["score"]

    if not score:
        logger.info("there is no score available yet, please upload the image and click on predict...")
        return context, "", "", update_log()
    
    threshold = context["threshold"][flavor]
    logger.info(f"apply threshold {threshold} for operation {flavor}...")
    recomendation = "Has tuberculoses" if score>threshold else "Not tuberculoses"
    return context, str(score), recomendation, update_log()

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
                with gr.Group():
                    flavor       = gr.Dropdown( flavors, value=default_flavor, label="Select the model operation:" )
                    with gr.Row():
                        score         = gr.Textbox(label="Model score:")
                        recomendation = gr.Textbox(label="Model recomendation:")

        with gr.Group():
            log = gr.Textbox(update_log(),label='data logger', max_lines=10)
            tag = gr.Textbox(model_tag, label='Model version:')

        # events
        predict_btn.click( predict , [context, flavor, image], [context, score, recomendation, log])
        flavor.change( change_flavor, [context, flavor], [context, score, recomendation, log])


#
# ************************************************************************************
#                                     Main Loop
# ************************************************************************************
#


def get_context():
    context = {
       "model"     : model,
       "preproc"   : preproc,
       "threshold" : threshold,
       "score"     : None,
    }
    return context

with gr.Blocks(theme="freddyaboulton/test-blue") as demo:
    context  = gr.State(get_context())
    gr.Label(f"CAD System (Amostra Gratis)", show_label=False)
    inference_tab(context, name='Inference')
    gr.Label("An initiative of the BRICS/UFRJ groups." , show_label=False)


# create APP
app = FastAPI()
app = gr.mount_gradio_app(app, demo, path="/inference")



if __name__ == "__main__":
    logger.info("Starting server...")
    demo.launch(server_port = 9000, share=False)
   
