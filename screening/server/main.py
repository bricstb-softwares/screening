

import gradio as gr
from fastapi import FastAPI
from loguru import logger

import tensorflow as tf
import pickle
import json

from loguru import logger

import matplotlib.pyplot as plt
model_from_json = tf.keras.models.model_from_json
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices)>0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.run_functions_eagerly(False)

class Inference:

    def __init__(self):
        pass

    def __call__(self, inputs ):
        pass

    def predict(self, inputs ):
        pass

    def load( self, path ):
        
        logger.info(f"reading model from {path}")

        with open(path, 'rb') as f:

            d = pickle.load(f)
            metadata = d['metadata']
            model = d["model"]
            logger.info("creating model...")
            model = model_from_json( json.dumps( d['model']['sequence'], separators=(',',':')) )
            model.set_weights( d['model']['weights'] )
            model.summary()
            self.model = model




# global inference handler
model = Inference()


def get_context():
    return {}

def process_data( context, data_path):
    model.load( data_path.name )
    return context



def config_tab( context, name = "Configuration"):

    with gr.Tab( name ):
        with gr.Row():
            with gr.Column():
                with gr.Group():
                    gpu_name = gr.Textbox( "", label="GPU")
                    gpu_memo = gr.Textbox( "", label="GPU Memory")
                    sys_memo = gr.Textbox( "", label="System Memory")
                    update_btn = gr.Button("Update")

            with gr.Column():
                with gr.Group():
                    upload_data = gr.UploadButton(label="Upload a pkl file with the model", file_types=[".pkl"])
                    model_hash  = gr.Textbox( "", label="Model hash")
                    model_version =  gr.Textbox( "", label="Model version")
    #with gr.Row():
    # actions
    upload_data.upload(fn=process_data, inputs=[context, upload_data], outputs=[context])



with gr.Blocks(theme='freddyaboulton/test-blue') as demo:
    gr.Label("", show_label=False)
    context = gr.State(get_context())
    config_tab(context, name = 'Model')
    gr.Label("", show_label=False)
    
    
app = FastAPI()
app = gr.mount_gradio_app(app, demo, path='/inference')




#@app.on_event("startup")
#async def startup_event():
#    consumer.start()
#
#@app.get("/runner/start") 
#async def start() -> schemas.Answer:
#    consumer.start()
#    return schemas.Answer( host=consumer.host_url, message="runner was started by external signal." )
#
#@app.get("/runner/ping")
#async def ping() -> schemas.Answer:
#    return schemas.Answer( host=consumer.host_url, message="pong" )






if __name__ == "__main__":
    logger.info("starting server...")
    demo.launch(share=False, server_port=3000, server_name="0.0.0.0")#, server_port=7860)