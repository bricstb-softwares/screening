


import gradio as gr
import tensorflow as tf
import numpy as np

from loguru  import logger
from config import server_flags as flags
from config import update_log
from fastapi import FastAPI

import utils


intro = """

## OpenTB

A tuberculose (TB) continua a ser uma das principais causas de mortalidade em países subdesenvolvidos, 
onde a infraestrutura de saúde frequentemente carece de recursos e tecnologias avançadas para diagnóstico 
precoce e tratamento eficaz. No Brasil, a situação é particularmente grave, com altas taxas de incidência 
e mortalidade, especialmente em áreas urbanas densamente povoadas como a cidade do Rio de Janeiro.
A complexidade do diagnóstico da tuberculose, que pode ser facilmente confundida com outras doenças 
pulmonares, agrava a situação, resultando em atrasos no tratamento e aumento da transmissão da doença.

Para enfrentar esse desafio, o Sistema OpenTB foi desenvolvido como um sistema de Detecção Assistida 
por Computador (CAD) para infecções por tuberculose. Este sistema utiliza inteligência artificial para 
aprimorar a precisão e a rapidez do diagnóstico, auxiliando os profissionais de saúde na identificação 
de casos de TB, especialmente em estágios iniciais.

O OpenTB baseia-se em dois modelos principais de inteligência artificial: redes neurais 
convolucionais (CNNs) e Redes Adversariais Generativas (WGANs). As CNNs são um tipo de rede neural 
profunda que são particularmente eficazes para tarefas de visão computacional. Elas funcionam aplicando 
filtros convolucionais sobre as imagens para detectar padrões e características específicas, como lesões
 pulmonares típicas da tuberculose em radiografias de tórax. A capacidade das CNNs de aprender a partir 
 de grandes conjuntos de dados de imagens permite que o sistema OpenTB identifique anomalias com alta 
 precisão, mesmo em casos sutis que podem ser difíceis de detectar para o olho humano.

As Redes Adversariais Generativas (WGANs) são outra classe de modelos utilizados no OpenTB, focando em 
gerar imagens sintéticas realistas que podem ser usadas para melhorar o treinamento dos modelos de CNN. 
As WGANs consistem em duas redes neurais que competem entre si: uma rede geradora que cria imagens falsas 
e uma rede discriminadora que tenta distinguir entre imagens reais e geradas. Esse processo adversarial 
resulta em um aprimoramento contínuo da qualidade das imagens geradas, fornecendo dados adicionais valiosos 
para treinar as CNNs e, assim, aumentar a precisão do diagnóstico do sistema.

O OpenTB é servido no Laboratório de Processamento de Sinais (LPS) da Universidade Federal do Rio de 
Janeiro (UFRJ), localizado na Ilha do Fundão. Este laboratório está equipado com infraestrutura de 
ponta para pesquisa e desenvolvimento em processamento de sinais e inteligência artificial. 
Através do LPS, o sistema OpenTB pode ser acessado remotamente por profissionais de saúde de qualquer 
parte do Rio de Janeiro e do Brasil, permitindo um diagnóstico rápido e eficiente, independentemente 
da localização geográfica.

Em resumo, o Sistema OpenTB representa uma avançada ferramenta tecnológica que pode revolucionar o 
combate à tuberculose em áreas subdesenvolvidas, fornecendo um diagnóstico preciso e acessível que 
é crucial para o controle e erradicação da doença.
"""



#
# NOTE: configure the GPU card
#
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices)>0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.run_functions_eagerly(False)



default_model_path="/mnt/brics_data/models/v1/user.philipp.gaspar.convnets_v1.baseline.shenzhen_santacasa.exp.20240303.r1/job.test_0.sort_0/output.pkl"

#
# This should be global to all sessions
#
model, preproc, threshold, model_tag = utils.load_model( default_model_path )
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
    gr.Label(f"OpenTB - LPS/UFRJ", show_label=False)
    inference_tab(context, name='Inference')
    gr.Markdown(intro , show_label=False)


# create APP
app = FastAPI()
app = gr.mount_gradio_app(app, demo, path="/")



if __name__ == "__main__":
    import uvicorn
    utils.setup_logs()
    uvicorn.run(app, host='0.0.0.0', port=9000, reload=False, log_level="warning")

