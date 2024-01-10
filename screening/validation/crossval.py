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
from pybeamer import *

import matplotlib.pyplot as plt
model_from_json = tf.keras.models.model_from_json
import atlas_mpl_style as ampl
ampl.use_atlas_style()


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




class crossval_table:
    #
    # Constructor
    #
    def __init__(self, config_dict):
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
        paths = expand_folders(path, filters=['*.pkl'])
        logger.info( f"Reading file for {tag} tag from {path}" )

        # Creating the dataframe
        dataframe = collections.OrderedDict({
                              'train_tag'      : [],
                              'op_name'        : [],
                              'test'           : [],
                              'sort'           : [],
                              'file_name'      : [],
                          })


        logger.info( f'There are {len(paths)} files for this task...')
        logger.info( f'Filling the table... ')

        for ituned_file_name in tqdm( paths , desc='Reading %s...'%tag):

            try:
                ituned = load_file(ituned_file_name)
            except:
                logger.fatal( f"File {ituned_file_name} not open. skip.")
                continue


            history = ituned['history']
            metadata = ituned['metadata'] 

            for op, config_dict in self.__config_dict.items():

                dataframe['train_tag'].append(tag)
                dataframe['file_name'].append(ituned_file_name)
                # get the basic from model
                dataframe['op_name'].append(op)
                dataframe['sort'].append(metadata['sort'])
                dataframe['test'].append(metadata['test'])

                # Get the value for each wanted key passed by the user in the contructor args.
                for key, local  in config_dict.items():
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
    def filter_sorts(self, key, idxmin=False):
        '''
        This method will filter the Dataframe based on given key in order to get the best inits for every sort.

        Arguments:

        - key: the column to be used for filter.
        '''
        if idxmin:
            idxmask = self.table.groupby(['train_tag', 'op_name', 'test'])[key].idxmin().values
            return self.table.loc[idxmask]
        else:
            idxmask = self.table.groupby(['train_tag', 'op_name','test'])[key].idxmax().values
            return self.table.loc[idxmask]


    #
    # Get the best sorts from best inits table
    #
    def filter_tests(self, best_sorts, key, idxmin=False):
        '''
        This method will filter the Dataframe based on given key in order to get the best model for every configuration.

        Arguments:

        - key: the column to be used for filter.
        '''
        if idxmin:
            idxmask = best_sorts.groupby(['train_tag', 'op_name'])[key].idxmin().values
            return best_sorts.loc[idxmask]
        else:
            idxmask = best_sorts.groupby(['train_tag', 'op_name'])[key].idxmax().values
            return best_sorts.loc[idxmask]



    def describe( self, best_sorts , 
                  exclude_keys = ['train_tag', 'test', 'sort', 'file_name', 'op_name','roc','roc_val','roc_op','roc_test']
                 ):

        dataframe = collections.OrderedDict({})
        def add(d, key,value):
            if key in d.keys():
                d[key].append(value)
            else:
                d[key] = [value]

        for train_tag in best_sorts.train_tag.unique():

            for op_name in best_sorts.op_name.unique():

                data = best_sorts.loc[ (best_sorts.train_tag==train_tag) & (best_sorts.op_name==op_name) ]
                add( dataframe , 'train_tag'      , train_tag             )
                add( dataframe , 'op_name'        , op_name               )
                for col_name in data.columns.values:
                    if col_name in exclude_keys:
                        continue    
                    add( dataframe , col_name +'_mean', data[col_name].mean() )
                    add( dataframe , col_name +'_std' , data[col_name].std()  )


        return pd.DataFrame(dataframe)


  

    def plot_roc_curve( self, best_sorts , best_sort,  output, title='', key=''):

        # preparation
        rocs_interp = []
        for roc in best_sorts[key].values:
            mean_fpr = np.linspace(0, 1, 100)
            interp_tpr = np.interp(mean_fpr, roc["fpr"], roc["tpr"]); interp_tpr[0] = 0.0
            rocs_interp.append(interp_tpr)
        
        mean_fpr    = np.linspace(0, 1, 100)
        mean_tpr    = np.mean(rocs_interp, axis=0); mean_tpr[-1] = 1.0
        std_tpr     = np.std(rocs_interp, axis=0)
        upper_error = mean_tpr + std_tpr
        lower_error = mean_tpr - std_tpr

      
        best_tpr = rocs_interp[best_sort.test.values[0]]
        best_fpr = mean_fpr


        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
        plt.fill_between( mean_fpr, lower_error, upper_error, label="Uncertainty", color="lightcoral" )
        plt.plot(mean_fpr, mean_tpr, color="blue", label="Mean ROC", alpha=0.8, linewidth=2)
        plt.vlines(0.3, 0, 1, colors="red", alpha=0.6, linestyles="--")
        plt.hlines(0.9, 0, 1, colors="red", alpha=0.6, linestyles="--")
        plt.plot( best_fpr, best_tpr, color='red' , label='Best', linewidth=2)

        
        
        axs.spines["left"].set_position(("outward", 10))
        axs.spines["top"].set_visible(False)
        axs.spines["right"].set_visible(False)
        axs.get_xaxis().tick_bottom()
        axs.get_yaxis().tick_left()
        plt.title( title )
        plt.ylabel("TPR (Sensitivity)")
        plt.xlabel("FPR (1 - Specificity)")

        plt.xlim(0.0, 0.7)
        plt.ylim(0.7, 1.0)
        plt.legend(loc="lower right")
        plt.savefig(output, bbox_inches="tight")
        plt.close()
        return output
    





    def report( self, best_sorts , best_models, title : str , outputFile : str , color_map = {}):


        cv_table    = self.describe( best_sorts )
        #best_models = self.best_models( best_sorts )

        # Default colors
        color = '\\cellcolor[HTML]{9AFF99}'

        color_map = {'tight':2,'medium':1,'loose':0}
     
        def colorize( row , op_name, color, color_map ):
            if op_name in color_map.keys():
                color_idx = color_map[op_name]
                row[color_idx] = color + row[color_idx]



        # Apply beamer
        with BeamerTexReportTemplate1( theme = 'Berlin'
                                       , _toPDF = True
                                       , title = title
                                       , outputFile = outputFile
                                       , font = 'structurebold' ):


            for op_name in cv_table.op_name.unique():


                with BeamerSection( name = ''):

                    color_col = color_map[op_name]

             
                    #
                    # For each train tag
                    #
                    for train_tag in cv_table.train_tag.unique():


                        _best_sorts = best_sorts.loc[(best_sorts.train_tag==train_tag)&(best_sorts.op_name==op_name)]
                        _best_model = best_models.loc[(best_models.train_tag==train_tag) & (best_models.op_name==op_name)]

                        #
                        # ROCs
                        #
                        figures = [
                            self.plot_roc_curve( _best_sorts, _best_model, f'roc_{train_tag}_{op_name}.pdf'      , key='roc'     , title='Train'),
                            self.plot_roc_curve( _best_sorts, _best_model, f'roc_{train_tag}_{op_name}_val.pdf'  , key='roc_val' , title='Val.' ),
                            self.plot_roc_curve( _best_sorts, _best_model, f'roc_{train_tag}_{op_name}_test.pdf' , key='roc_test', title='Test' ),
                        ]


                        BeamerMultiFigureSlide( title = f'{train_tag} ROC curves for {train_tag} ({op_name})'
                            , paths = figures
                            , nDivWidth = 3 # x
                            , nDivHeight = 1 # y
                            , texts=None
                            , fortran = False
                            , usedHeight = 0.4
                            , usedWidth  = 1
                            )

                        t = cv_table.loc[ (cv_table.train_tag==train_tag) & (cv_table.op_name==op_name) ]


                        #
                        # Tables
                        #

                        lines1 = []
                        lines1 += [ HLine(_contextManaged = False) ]
                        lines1 += [ HLine(_contextManaged = False) ]


                        col_names =  [r'Sens. [\%]', r'SP [\%]', r'Spec. [\%]', r'AUC [\%]']
                        colorize( col_names, op_name, color, color_map)

                        lines1 += [ TableLine( columns = ['', '']+col_names, _contextManaged = False ) ]
                        lines1 += [ HLine(_contextManaged = False) ]


                        keys    = ['sens_at', 'sp_index', 'spec_at' , 'auc']
                        row     = [ '%1.2f$\pm$%1.2f'%(t[key+'_mean']*100 , t[key+'_std']*100) for key in keys]
                        colorize( row, op_name, color, color_map)
                        lines1 += [ TableLine( columns = ['\multirow{4}{*}{'+train_tag+'}', 'Train']  + row , _contextManaged = False ) ]

                        keys    = ['sens_at_val', 'sp_index_val', 'spec_at_val', 'auc_val' ]
                        row     = [ '%1.2f$\pm$%1.2f'%(t[key+'_mean']*100 , t[key+'_std']*100) for key in keys]
                        colorize( row, op_name, color, color_map)
                        lines1 += [ TableLine( columns = ['', 'Val.']  + row , _contextManaged = False ) ]

                        keys    = ['sens_at_test', 'sp_index_test', 'spec_at_test' , 'auc_test']
                        row     = [ '%1.2f$\pm$%1.2f'%(t[key+'_mean']*100 , t[key+'_std']*100) for key in keys]
                        colorize( row, op_name, color, color_map)
                        lines1 += [ TableLine( columns = ['', 'Test']  + row , _contextManaged = False ) ]

                        t = best_models.loc[ (best_models.train_tag==train_tag) & (best_models.op_name==op_name) ]
                        keys    = ['sens_at_test', 'sp_index_test', 'spec_at_test' , 'auc_test' ]
                        row     = [ '%1.2f'%(t[key]*100) for key in keys]
                        colorize( row, op_name, color, color_map)
                        lines1 += [ TableLine( columns = ['', 'Best']  + row , _contextManaged = False ) ]

                        lines1 += [ HLine(_contextManaged = False) ]
                        lines1 += [ HLine(_contextManaged = False) ]

                        # Create all tables into the PDF Latex
                        with BeamerSlide( title = f"The Cross Validation: {train_tag} ({op_name})"  ):
                            with Table( caption = f'The values for each set using {train_tag} method.') as table:
                                with ResizeBox( size = 1. ) as rb:
                                    with Tabular( columns = '|lc|' + 'cccc|' ) as tabular:
                                        tabular = tabular
                                        for line in lines1:
                                            if isinstance(line, TableLine):
                                                tabular += line
                                            else:
                                                TableLine(line, rounding = None)


                    #
                    # Compare train tags
                    #
                    col_names =  [r'Sens. [\%]', r'SP [\%]', r'Spec. [\%]', r'AUC [\%]']
                    colorize( col_names, op_name, color, color_map)

                    lines1 = []
                    lines1 += [ HLine(_contextManaged = False) ]
                    lines1 += [ HLine(_contextManaged = False) ]   
                    lines1 += [ TableLine( columns = ['', '',] + col_names, _contextManaged = False ) ]
                    lines1 += [ HLine(_contextManaged = False) ]   

                    for train_tag in cv_table.train_tag.unique():
                        t = cv_table.loc[ (cv_table.train_tag==train_tag) & (cv_table.op_name==op_name) ]
                        keys    = ['sens_at_test', 'sp_index_test', 'spec_at_test' , 'auc_test']
                        row     = [ '%1.2f$\pm$%1.2f'%(t[key+'_mean']*100 , t[key+'_std']*100) for key in keys]
                        colorize( row, op_name, color, color_map)
                        lines1 += [ TableLine( columns = [ train_tag, 'Test']  + row , _contextManaged = False ) ]

                    lines1 += [ HLine(_contextManaged = False) ]
                    lines1 += [ HLine(_contextManaged = False) ]

                    # Create all tables into the PDF Latex
                    with BeamerSlide( title = f"The Cross Validation: {op_name}"  ):
                        with Table( caption = r'The values for each method.') as table:
                            with ResizeBox( size = 1. ) as rb:
                                with Tabular( columns = '|lc|' + 'cccc|' ) as tabular:
                                    tabular = tabular
                                    for line in lines1:
                                        if isinstance(line, TableLine):
                                            tabular += line
                                        else:
                                            TableLine(line, rounding = None)

               