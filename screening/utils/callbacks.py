__all__ = []



import os, sys, json, pickle
import tensorflow as tf

from tensorflow.keras.models import load_model
from loguru import logger

#
# custom callbacks
#

class MinimumEpochs(tf.keras.callbacks.Callback):
    def __init__(self, min_epochs : int = 10):
        super(MinimumEpochs, self).__init__()
        self.min_epochs = min_epochs

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.min_epochs - 1:
            self.model.stop_training = False



class EarlyStopping(tf.keras.callbacks.Callback):

    def __init__(self, monitor: str='var_loss', 
                       mode : str='min' , 
                       patience : int=10,
                       checkpoint_path : str=os.getcwd(),
                       do_checkpoint : bool=True,
                       ):
        
        super(EarlyStopping, self).__init__()
        self.monitor             = monitor
        self.mode                = mode
        self.patience            = patience
        self.patience_count      = 0
        self.do_checkpoint       = do_checkpoint
        self.checkpoint_path     = checkpoint_path
        self.checkpoint_filepath = checkpoint_path+'/checkpoint.json'

        if self.do_checkpoint and os.path.exists(self.checkpoint_filepath):  
            logger.info("loading from last checkpoint...")
            with open(self.checkpoint_filepath,'r') as f:
                d = json.load(f)
                model_filepath     = d['model_path']
                model              = load_model(model_filepath)
                self.best_weights  = model.get_weights()
                history_filepath   = d['history_path']
                self.best_loss     = d['best_loss']
                self.initial_epoch = d['initial_epoch']
                self.best_epoch    = d['best_epoch']
                with open(history_filepath,'rb') as h:
                    history = pickle.load(h)['history']
                    self.last_history = history
        else:
            logger.info("starting from begginer...")
            self.last_history  = {}
            self.best_epoch    = 0
            self.best_loss     = sys.float_info.max if mode=='min' else sys.float_info.min
            self.initial_epoch = 0
            self.best_weights  = None




    def on_epoch_end(self, epoch, logs=None):

        self.update_history(logs)

        loss=round(logs[self.monitor],4)
        
        
        if (self.mode=='min' and loss<self.best_loss) or (self.mode=='max' and loss>self.best_loss):

            logger.info(f"best ({self.monitor}) reached ({loss}) for epoch {epoch+1}, reset patience count...")
            self.best_loss=loss   
            self.patience_count=0
            self.best_epoch=epoch
            self.best_weights = self.model.get_weights()

            if self.do_checkpoint:
                logger.info(f"save current stage into {self.checkpoint_filepath}...")

                model_filepath=self.checkpoint_path+'/best_model.h5'
                history_filepath=self.checkpoint_path+'/history.pkl'
                temp_model_filepath=self.checkpoint_path+'/.best_model.h5'
                temp_history_filepath=self.checkpoint_path+'/.history.pkl'

                # saving...
                self.model.save(temp_model_filepath)
                with open(temp_history_filepath,'wb') as f:
                    pickle.dump({'history':self.last_history}, f)

                # create checkpoint form....
                with open(self.checkpoint_path+"/checkpoint.json",'w') as f:
                    d = {
                        'initial_epoch' : epoch + 1, # index fix since keras epochs starts from 1...
                        'best_loss'     : self.best_loss,
                        'model_path'    : model_filepath,
                        'history_path'  : history_filepath,
                        'best_epoch'    : self.best_epoch,
                        }
                    json.dump(d,f)
                os.rename( temp_model_filepath  , model_filepath   )
                os.rename( temp_history_filepath, history_filepath )
        else:
            self.patience_count+=1


        if self.patience_count > self.patience:
            logger.info("stop training...")
            self.model.stop_training = True




    def on_train_end(self, logs={}):
        # Loading the best model
        if self.best_weights:
            logger.info("reloading best weights into the model...")
            self.model.set_weights( self.best_weights )
        logger.info(f"reloading history and merge...")
        self.model.history.history=self.last_history
        history=self.model.history.history
        logger.info(f"final traning has {len(history[self.monitor])} epochs after history merge...")



    def update_history(self,logs):
        for key, value in logs.items():
            if key in self.last_history.keys():
                self.last_history[key].append(value)
            else:
                self.last_history[key]=[value]




#
# load checkpoint
#
def load_model_from_checkpoint(  checkpoint_path : str=os.getcwd() ):
    with open(checkpoint_path+"/checkpoint.json",'r') as f:
        checkpoint     = json.load(f)
        model          = load_model( checkpoint['model_path'] )
        return model

