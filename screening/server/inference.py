


from loguru import logger

import matplotlib.pyplot as plt
model_from_json = tf.keras.models.model_from_json


class Inference:

    def __init__(self):



    def __call__(self, )


    def predict(self, ):


    def load( self, path ):

        with open(path, 'rf') as f:

            logger.info(f"reading model from {path}")
            d = pickle.load(f)
            
            metadata = d['metadata']
            model = d["model"]
            logger.info("creating model...")
            model = model_from_json( json.dumps( d['model']['sequence'], separators=(',',':')) )
            model.set_weights( d['model']['weights'] )
            model.summary()
            self.model = model


