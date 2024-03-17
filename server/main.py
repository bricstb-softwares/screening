


import gradio as gr
import tensorflow as tf
import numpy as np

from loguru  import logger
from config import server_flags as flags
from config import update_log
from fastapi import FastAPI

import utils



#
# NOTE: configure the GPU card
#
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices)>0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.run_functions_eagerly(False)




#
# This should be global to all sessions
#
model, preproc, threshold, model_tag = utils.load_model( flags.model )
flavors        = list(threshold.keys())
default_flavor = flavors[0]
    


#
# events 
#



def predict( context, flavor):

    uploaded = context["uploaded"]
    if not uploaded:
        logger.warning("there is no image inside. please upload the image first and click on predict.")
        return context, "", "", gr.Image( value=None, type='numpy'), update_log()

    threshold = 0.1
    logger.info("predicting....")    
    model     = context["model"]
    threshold = context["flavors"][flavor]
    image     = context['image']
    logger.info(f"operation point is {flavor}")
    ## NOTE: predict
    img = np.expand_dims(image, 0)
    score = model.predict( img )[0][0] # NOTE: get single value since this is a one sample mode
    logger.info(f"output prob as {score}")
    recomendation = "Not normal" if score>threshold else "Normal"
    logger.info(f"model recomentation is : {recomendation}")
    saliency = utils.get_saliency(model, image, )
    image = utils.paint_saliency( image, saliency, threshold)

    context['saliency_threshold'] = threshold 
    context["score"]         = score
    context["saliency"]      = saliency
    context["recomendation"] = recomendation
    context["predicted"]     = True
    return context, str(score), recomendation, image, update_log()


def change_importance( context, slider):

    predicted = context["predicted"]
    if not predicted:
        logger.info("there is no prediction available yet, please upload the image and click on predict...")
        return context, gr.Image( type='numpy'), update_log()

    threshold = slider
    image     = context['image']
    saliency  = context['saliency']
    context["saliency_threshold"] = threshold
    image = utils.paint_saliency(image,saliency, threshold)
    return context, image, update_log()


def change_flavor(context, flavor):

    predicted = context["predicted"]
    if not predicted:
        logger.info("there is no prediction available yet, please upload the image and click on predict...")
        return context, "", "", update_log()
    
    score = context["score"]     
    threshold = context["flavors"][flavor]
    logger.info(f"apply threshold {threshold} for operation {flavor}...")
    recomendation = "Not normal" if score>threshold else "Normal"
    return context, str(score), recomendation, update_log()



def upload( context , image_path , auto_crop):
    
    context["predicted"]=False

    logger.info("applying preprocessing...")
    if not image_path:
        context["uploaded"]=False
        logger.info("reseting everything....")
        return context, gr.Image(value=None, type='numpy'), update_log()


    logger.info(image_path)
    context["uploaded"]=True
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
                    image       = gr.Image(label='upload', type='filepath' )
                    auto_crop   = gr.Checkbox(label="auto cropping", info="auto cropping for digital images", value=True)

            with gr.Column():

                with gr.Group():
                    image_processed = gr.Image(show_label=False,show_download_button=False,label='image display', type='numpy')
                    importance = gr.Slider(0, 1, value=0.7, label="Importance", info="Choose between 0 and 1")
                    with gr.Row():
                        flavor     = gr.Dropdown( flavors, value=default_flavor, label="select the model operation:" )
                    with gr.Row():
                        score         = gr.Textbox(label="model score:")
                        recomendation = gr.Textbox(label="model recomendation:")
                    predict_btn = gr.Button("predict")


        with gr.Accordion(open=False, label='detailed information'):
            with gr.Group():
                tag = gr.Textbox(model_tag, label='model version:')
                log = gr.Textbox(update_log(),label='logger', max_lines=10,lines=10)

        # events
        image.change( upload , [context, image, auto_crop], [context , image_processed, log])
        flavor.change( change_flavor, [context, flavor], [context, score, recomendation, log])

        predict_btn.click( predict , [context, flavor], [context, score, recomendation, image_processed, log]) 
        importance.change( change_importance, [context, importance], [context,image_processed,log])

#
# ************************************************************************************
#                                     Main Loop
# ************************************************************************************
#


def get_context():
    context = {
       "model"     : model,
       "preproc"   : preproc,
       "predicted" : False,
       "uploaded"  : False,
       "flavors"   : threshold,
    }
    return context

with gr.Blocks(theme="freddyaboulton/test-blue") as demo:
    context  = gr.State(get_context())
    gr.Label(f"CAD UFRJ", show_label=False)
    inference_tab(context, name='Inference')
    gr.Label("An initiative of the BRICS/UFRJ group." , show_label=False)


# create APP
app = FastAPI()
app = gr.mount_gradio_app(app, demo, path="/inference")



if __name__ == "__main__":
    import uvicorn
    utils.setup_logs()
    uvicorn.run(app, host='0.0.0.0', port=9000, reload=False, log_level="warning")

