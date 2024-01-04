__all__ = ["crossval_table"]

from pprint import pprint
from itertools import product
from tqdm import tqdm
from loguru import logger
from sklearn.metrics import auc, confusion_matrix, roc_curve
from expand_folders import expand_folders

import collections, os, glob, json, copy, re, pickle
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import screening.utils.convnets as convnets

model_from_json = tf.keras.models.model_from_json



def load_file( path ):
  with open(path, 'rb') as f:
    d = pickle.load(f)
    return d

def get_nn_model(path):
    d = load_file(path)
    model = model_from_json( json.dumps( d['model'], separators=(',',':')) )
    model.set_weights( d['weights'] )
    metadata = d['metadata']
    return model, metadata


# Cross validation reader configuration
cv_info = collections.OrderedDict( {
              "max_sp_val"      : 'summary/max_sp',
              "max_sp_pd"       : 'summary/max_sp_pd',
       
              } )


class crossval_table:
    #
    # Constructor
    #
    def __init__(self, config_dict=cv_info):
        '''
        The objective of this class is extract the tuning information from saphyra's output and
        create a pandas DataFrame using then.
        The informations used in this DataFrame are listed in info_dict, but the user can add more
        information from saphyra summary for example.


        Arguments:

        - config_dict: a dictionary contains in keys the measures that user want to check and
        the values need to be a empty list.

        Ex.: info = collections.OrderedDict( {

              "max_sp_val"      : 'summary/max_sp_val',
              "max_sp_pd_val"   : 'summary/max_sp_pd_val',
              "max_sp_fa_val"   : 'summary/max_sp_fa_val',
              "op_max_sp"       : 'summary/op_max_sp',
              "op_max_sp_pd"    : 'summary/op_max_sp_pd',
              "op_max_sp_fa"    : 'summary/op_max_sp_fa',
              "val_max_sp"      : 'summary/val_max_sp',
              "val_max_sp_pd"   : 'summary/val_max_sp_pd',
              "val_max_sp_fa"   : 'summary/val_max_sp_fa',
              "max_sp_spec"     : 'summary/max_sp_spec',
              "max_sp_sens"     : 'summary/max_sp_sens',
              "sens"            : 'summary/sens',
              "spec"            : 'summary/spec',
              "mse"             : 'summary/mse',
              "auc"             : 'summary/auc',
              } )

        - etbins: a list of et bins edges used in training;
        - etabins: a list of eta bins edges used in training;
        '''
        # Check wanted key type
        self.__config_dict = collections.OrderedDict(config_dict) if type(config_dict) is dict else config_dict
        self.table = None


    #
    # Fill the main dataframe with values from the tuning files and convert to pandas dataframe
    #
    def fill(self, path, tag):
        '''
        This method will fill the information dictionary and convert then into a pandas DataFrame.

        Arguments.:

        - path: the path to the tuned files;
        - tag: the training tag used;
        '''
        paths = glob.glob(path)
        logger.info( f"Reading file for {tag} tag from {path}" )

        # Creating the dataframe
        dataframe = collections.OrderedDict({
                              'train_tag'      : [],
                              'model_idx'      : [],
                              'sort'           : [],
                              'init'           : [],
                              'file_name'      : [],
                          })


        logger.info( f'There are {len(paths)} files for this task...')
        logger.info( f'Filling the table... ')

        for ituned_file_name in tqdm( paths , desc=fire+' Reading %s...'%tag):

            try:
                ituned = load_file(ituned_file_name)
            except:
                logger.fatal( f"File {ituned_file_name} not open. skip.")
                continue


            history = ituned['history']
            metadata = ituned['metadata'] 
            dataframe['train_tag'].append(tag)
            dataframe['file_name'].append(ituned_file_name)
            # get the basic from model
            dataframe['model_idx'].append(metadata['model_idx'])
            dataframe['sort'].append(metadata['sort'])
            dataframe['init'].append(metadata['init'])

            # Get the value for each wanted key passed by the user in the contructor args.
            for key, local  in self.__config_dict.items():
                if not key in dataframe.keys():
                    dataframe[key] = [self.__get_value( history, local )]
                else:
                    dataframe[key].append( self.__get_value( history, local ) )
    
        # Loop over all files


        # append tables if is need
        # ignoring index to avoid duplicated entries in dataframe
        self.table = pd.concat( (self.table, pd.DataFrame(dataframe) ), ignore_index=True ) if self.table is not None else pd.DataFrame(dataframe)
        logger.info( 'End of fill step, a pandas DataFrame was created...')


    #
    # Convert the table to csv
    #
    def to_csv( self, output ):
        '''
        This function will save the pandas Dataframe into a csv file.

        Arguments.:

        - output: the path and the name to be use for save the table.

        Ex.:
        m_path = './my_awsome_path
        m_name = 'my_awsome_name.csv'

        output = os.path.join(m_path, m_name)
        '''
        self.table.to_csv(output, index=False)


    #
    # Read the table from csv
    #
    def from_csv( self, input ):
        '''
        This method is used to read a csv file insted to fill the Dataframe from tuned file.

        Arguments:

        - input: the csv file to be opened;
        '''
        self.table = pd.read_csv(input)



    #
    # Get the value using recursive dictionary navigation
    #
    def __get_value(self, history, local):
        '''
        This method will return a value given a history and dictionary with keys.

        Arguments:

        - history: the tuned information file;
        - local: the path caming from config_dict;
        '''
        # Protection to not override the history since this is a 'mutable' object
        var = copy.copy(history)
        for key in local.split('/'):
            var = var[key.split('#')[0]][int(key.split('#')[1])] if '#' in key else var[key]
        return var


    #
    # Return only best inits
    #
    def filter_inits(self, key, idxmin=False):
        '''
        This method will filter the Dataframe based on given key in order to get the best inits for every sort.

        Arguments:

        - key: the column to be used for filter.
        '''
        if idxmin:
            idxmask = self.table.groupby(['train_tag', 'model_idx', 'sort'])[key].idxmin().values
            return self.table.loc[idxmask]
        else:
            idxmask = self.table.groupby(['train_tag', 'model_idx', 'sort'])[key].idxmax().values
            return self.table.loc[idxmask]


    #
    # Get the best sorts from best inits table
    #
    def filter_sorts(self, best_inits, key, idxmin=False):
        '''
        This method will filter the Dataframe based on given key in order to get the best model for every configuration.

        Arguments:

        - key: the column to be used for filter.
        '''
        if idxmin:
            idxmask = best_inits.groupby(['train_tag', 'model_idx'])[key].idxmin().values
            return best_inits.loc[idxmask]
        else:
            idxmask = best_inits.groupby(['train_tag', 'model_idx'])[key].idxmax().values
            return best_inits.loc[idxmask]


    def filter_neurons( self, best_sorts, key, idxmin = False):
        if idxmin:
            idxmask = best_sorts.groupby(['train_tag'])[key].idxmin().values
            return best_sorts.loc[idxmask]
        else:
            idxmask = best_sorts.groupby(['train_tag'])[key].idxmax().values
            return best_sorts.loc[idxmask]


    def describe(self, best_inits):

        dataframe = {'train_tag', }
 

