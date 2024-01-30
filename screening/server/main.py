
import gradio as gr
import sys, os, pickle, json

from fastapi import FastAPI
from loguru import logger
from config import server_flags as flags
from inference import inference_tab, create_inference_vars




logger.add(flags.log_path,  rotation="50 MB", backtrace=True, diagnose=True)




def get_context():
    context = {}
    context.update(create_inference_vars())
    return context



#
# Main Loop
#
with gr.Blocks(theme="freddyaboulton/test-blue") as demo:
    context  = gr.State(get_context())
    gr.Label(f"Inference Server", show_label=False)
    inference_tab(context, name='Inference')
    gr.Label("An initiative of the BRICS/LPS group." , show_label=False)


# create APP
app = FastAPI()
app = gr.mount_gradio_app(app, demo, path="/inference")



if __name__ == "__main__":
    logger.info("Starting server...")
    demo.launch(server_port = 9000, share=False)
   
