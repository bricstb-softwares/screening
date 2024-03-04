


import gradio as gr
import sys, socket, pickle, json
import tensorflow as tf
import numpy as np
import collections

from loguru  import logger
from config import server_flags as flags
from config import update_log
from fastapi import FastAPI
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils import normalize
from vis.utils import utils



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

setup_logs()


#
# NOTE: configure the GPU card
#
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices)>0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.run_functions_eagerly(False)



def get_saliency( model, image ):
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


#
# This should be global to all sessions
#
model, preproc, threshold, model_tag = load_model( flags.model )
flavors        = list(threshold.keys())
default_flavor = flavors[0]
    


#
# events 
#



def predict( context, flavor):


    logger.info("predicting....")
    
    model     = context["model"]
    threshold = context["threshold"][flavor]
    image     = context['image']

    logger.info(f"operation point is {flavor}")
    ## NOTE: predict
    img = np.expand_dims(image, 0)
    score = model.predict( img )[0][0] # NOTE: get single value since this is a one sample mode
    logger.info(f"output prob as {score}")
    context["score"] = score
    recomendation = "Not normal" if score>threshold else "Normal"
    logger.info(f"model recomentation is : {recomendation}")


    image = get_saliency(model, image, )

    return context, str(score), recomendation, image, update_log()


def change_flavor(context, flavor):
    score = context["score"]

    if not score:
        logger.info("there is no score available yet, please upload the image and click on predict...")
        return context, "", "", update_log()
    
    threshold = context["threshold"][flavor]
    logger.info(f"apply threshold {threshold} for operation {flavor}...")
    recomendation = "Not normal" if score>threshold else "Normal"
    return context, str(score), recomendation, update_log()



def apply_preproc_image( context , image_path , auto_crop):

    logger.info("applying preprocessing...")
    image=preproc(image_path, crop=auto_crop)
    context["image"]=image
    return context, image, update_log()




#
# NOTE: Build labelling tab here with all trigger functions inside
#
def inference_tab(context ,  name = 'Inference'):


    with gr.Tab(name):
        with gr.Row():
            with gr.Column():
                with gr.Group():
                    image       = gr.Image(label='', 
                                                 type='filepath' , 
                                                 #show_download_button=True,
                                                 #interactive=True,
                                                 #height=600, width=600
                                                 )
                    auto_crop   = gr.Checkbox(label="auto cropping", info="Auto cropping for digital images", value=True)

            with gr.Column():

                with gr.Group():
                    flavor       = gr.Dropdown( flavors, value=default_flavor, label="Select the model operation:" )
                    with gr.Row():
                        score         = gr.Textbox(label="Model score:")
                        recomendation = gr.Textbox(label="Model recomendation:")
                    predict_btn = gr.Button("Predict")

                with gr.Accordion(open=False, label='Detailed information'):
                    with gr.Row():
                        image_processed = gr.Image(label='processed image', type='numpy', height=256, width=256)
                        image_saliency  = gr.Image(label='saliency image', type='numpy', height=256, width=256)

        with gr.Group():
            log = gr.Textbox(update_log(),label='data logger', max_lines=10)
            #tag = gr.Textbox(model_tag, label='Model version:')

        # events

        image.change( apply_preproc_image , [context, image, auto_crop], [context , image_processed, log])
        predict_btn.click( predict , [context, flavor], [context, score, recomendation, image_saliency, log]) 
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
    gr.Label(f"CAD", show_label=False)
    inference_tab(context, name='Inference')
    gr.Label("An initiative of the BRICS/UFRJ group." , show_label=False)


# create APP
app = FastAPI()
app = gr.mount_gradio_app(app, demo, path="/inference")



if __name__ == "__main__":
    import uvicorn
    #logger.info("Starting server...")
    uvicorn.run(app, host='0.0.0.0', port=9000, reload=False, log_level="warning")

