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
    def filter_sorts(self, table, key, idxmin=False):
        '''
        This method will filter the Dataframe based on given key in order to get the best inits for every sort.

        Arguments:

        - key: the column to be used for filter.
        '''
        if idxmin:
            idxmask = table.groupby(['train_tag', 'op_name', 'test'])[key].idxmin().values
            return table.loc[idxmask]
        else:
            idxmask = table.groupby(['train_tag', 'op_name','test'])[key].idxmax().values
            return table.loc[idxmask]


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


    def get_out_of_sample( self, table ):

        dataframe = collections.OrderedDict()
        def add(d, key,value):
            if key in d.keys():
                d[key].append(value)
            else:
                d[key] = [value]   
        for idx, row in table.iterrows():
            
            for dataset_name in row.inference.keys():
            
                add( dataframe, 'train_tag', row.train_tag )
                add( dataframe, 'op_name'  , row.op_name   )
                add( dataframe, 'dataset_name' , dataset_name)
                d = row.inference[dataset_name]
                for col, value in d.items():
                    add( dataframe, col, value )
    
        return pd.DataFrame(dataframe)


    def describe( self, best_sorts , 
                  exclude_keys = ['train_tag', 'test', 'sort', 'file_name', 'op_name','roc','roc_val','roc_op','roc_test','inference']
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


  

    def plot_roc_curve( self, best_sorts , best_sort,  output, title='', key='', color='tomato'):

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
        plt.fill_between( mean_fpr, lower_error, upper_error, label="Uncertainty", color=color )
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
    

    def get_history(self, path):
        ituned = load_file(path)
        history = ituned['history']
        return history

	#
	# Plot the training curves for all sorts.
	#
    def plot_training_curves( self, table, best_sorts, basepath, display=False, start_epoch=0 ):
        '''
        This method is a shortcut to plot the monitoring traning curves.

        Arguments:

        - best_inits: a pandas Dataframe which contains all information for the best inits;
        - best_sorts: a pandas Dataframe which contains all information for the best sorts;
        - dirname: a folder to save the figures, if not exist we'll create and attached in $PWD folder;
        - display: a boolean to decide if show or not show the plot;
        - start_epoch: the epoch to start draw the plot.
        '''
        os.makedirs( basepath, exist_ok=True)

        def plot_training_curves_for_each_test(table, test, best_sort , output, display=False, start_epoch=0, ):

            table  = table.loc[table.test==test]
            nsorts = len(table.sort.unique())
            fig, ax = plt.subplots(5,2, figsize=(30,20))
            fig.suptitle(r'Monitoring Train Plot - Test = %d'%(test), fontsize=15)
            max_sort_per_column = 5
            col_idx = 0; row_idx = 0

            sorts = table.sort.unique()
            sorts = sorted(sorts)
            for sort in sorts:

                current_table = table.loc[table.sort==sort]
                path=current_table.file_name.values[0]
                history = self.get_history( path )
                
                best_epoch = history['best_epoch'] - start_epoch
                # Make the plot here
                ax[row_idx, col_idx].set_ylabel('Loss (sort = %d)'%sort, color = 'r' if best_sort==sort else 'k')
                ax[row_idx, col_idx].plot(history['loss'][start_epoch::], c='b', label='Train Step')
                ax[row_idx, col_idx].plot(history['val_loss'][start_epoch::], c='r', label='Validation Step')
                ax[row_idx, col_idx].axvline(x=best_epoch, c='k', label='Best epoch')
                ax[row_idx, col_idx].legend()
                ax[row_idx, col_idx].grid()
                
                if row_idx < max_sort_per_column - 1:
                    row_idx+=1
                else:
                    row_idx=0
                    col_idx+=1
            
            # ensure that last rows for each column are filled
            ax[4, 0].set_xlabel('Epochs')
            ax[4, 1].set_xlabel('Epochs')


            plt.savefig(output)
            if display:
                plt.show()
            else:
                plt.close(fig)

        figures = []
       
        for test in table.test.unique():
            train_tag = table.train_tag.values[0]; op_name = table.op_name.values[0]
            figurepath = basepath+'/train_evolution_%s_%s_test_%d.pdf'%(op_name, train_tag, test)
            best_sort  = best_sorts.loc[best_sorts.test==test].sort.values[0]
            plot_training_curves_for_each_test( table, test, best_sort, figurepath, start_epoch=start_epoch)
            figures.append(figurepath)

        return figures





    def report( self, table,  best_sorts , best_models, 
                title      : str , 
                outputFile : str , 
                detailed   : bool=True, 
                color_map  : dict={} ):

        color_map   = {'tight':2,'medium':1,'loose':0}
        cv_table    = self.describe( best_sorts )
        #best_models = self.best_models( best_sorts )

        # Default colors
        color = '\\cellcolor[HTML]{9AFF99}'

     
        def colorize( row , op_name, color, color_map ):
            if op_name in color_map.keys():
                color_idx = color_map[op_name]
                row[color_idx] = color + row[color_idx]

        basedir=os.getcwd()+'/figures'

        # Apply beamer
        with BeamerTexReportTemplate1( theme = 'Berlin'
                                       , _toPDF = True
                                       , title = title
                                       , outputFile = outputFile
                                       , font = 'structurebold' ):


            for op_name in cv_table.op_name.unique():


                with BeamerSection( name = op_name):

                    color_col = color_map[op_name]

             
                    #
                    # For each train tag
                    #
                    for train_tag in cv_table.train_tag.unique():

                        basepath = basedir+'/'+op_name+'/'+train_tag
                        os.makedirs( basepath, exist_ok=True)

                        _best_sorts = best_sorts.loc[(best_sorts.train_tag==train_tag)&(best_sorts.op_name==op_name)]
                        _best_model = best_models.loc[(best_models.train_tag==train_tag) & (best_models.op_name==op_name)]

                        if detailed:
                            _table      = table.loc[(table.train_tag==train_tag)&(table.op_name==op_name)]

                            #
                            # Train curves
                            #
                            figures = self.plot_training_curves( _table, _best_sorts, basepath )
                            for figure in figures:
                                BeamerFigureSlide( title = f'Training curves for {train_tag} in {op_name} operation.'
                                    , path = figure
                                    , texts=None
                                    , fortran = False
                                    , usedHeight = 1
                                    , usedWidth  = 1
                                    )

                            #
                            # ROCs
                            #
                            figures = [
                                self.plot_roc_curve( _best_sorts, _best_model, f'{basedir}/roc_{train_tag}_{op_name}.pdf'      , key='roc'     , title='Train'),
                                self.plot_roc_curve( _best_sorts, _best_model, f'{basedir}/roc_{train_tag}_{op_name}_val.pdf'  , key='roc_val' , title='Val.' ),
                                self.plot_roc_curve( _best_sorts, _best_model, f'{basedir}/roc_{train_tag}_{op_name}_test.pdf' , key='roc_test', title='Test' ),
                            ]
                            BeamerMultiFigureSlide( title = f'ROC curves for {train_tag} in {op_name} operation.'
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
                        lines = []
                        lines += [ HLine(_contextManaged = False) ]
                        lines += [ HLine(_contextManaged = False) ]
                        col_names =  [r'Sens. [\%]', r'SP [\%]', r'Spec. [\%]', r'AUC [\%]']
                        colorize( col_names, op_name, color, color_map)
                        lines += [ TableLine( columns = ['', '']+col_names, _contextManaged = False ) ]
                        lines += [ HLine(_contextManaged = False) ]
                        keys    = ['sens_at', 'sp_index', 'spec_at' , 'auc']
                        row     = [ '%1.2f$\pm$%1.2f'%(t[key+'_mean']*100 , t[key+'_std']*100) for key in keys]
                        colorize( row, op_name, color, color_map)
                        lines += [ TableLine( columns = ['\multirow{4}{*}{'+train_tag+'}', 'Train']  + row , _contextManaged = False ) ]
                        keys    = ['sens_at_val', 'sp_index_val', 'spec_at_val', 'auc_val' ]
                        row     = [ '%1.2f$\pm$%1.2f'%(t[key+'_mean']*100 , t[key+'_std']*100) for key in keys]
                        colorize( row, op_name, color, color_map)
                        lines += [ TableLine( columns = ['', 'Val.']  + row , _contextManaged = False ) ]
                        keys    = ['sens_at_test', 'sp_index_test', 'spec_at_test' , 'auc_test']
                        row     = [ '%1.2f$\pm$%1.2f'%(t[key+'_mean']*100 , t[key+'_std']*100) for key in keys]
                        colorize( row, op_name, color, color_map)
                        lines += [ TableLine( columns = ['', 'Test']  + row , _contextManaged = False ) ]
                        t = best_models.loc[ (best_models.train_tag==train_tag) & (best_models.op_name==op_name) ]
                        keys    = ['sens_at_test', 'sp_index_test', 'spec_at_test' , 'auc_test' ]
                        row     = [ '%1.2f'%(t[key]*100) for key in keys]
                        colorize( row, op_name, color, color_map)
                        lines += [ TableLine( columns = ['', 'Best (Test)']  + row , _contextManaged = False ) ]
                        lines += [ HLine(_contextManaged = False) ]
                        lines += [ HLine(_contextManaged = False) ]
                        # Create all tables into the PDF Latex
                        with BeamerSlide( title = f"The in-sample cross val. for {train_tag} in {op_name} operation."  ):
                            with Table( caption = f'The in-sample mean and standard dev. values for each set.') as table:
                                with ResizeBox( size = 1. ) as rb:
                                    with Tabular( columns = '|lc|' + 'cccc|' ) as tabular:
                                        tabular = tabular
                                        for line in lines:
                                            if isinstance(line, TableLine):
                                                tabular += line
                                            else:
                                                TableLine(line, rounding = None)



                        #
                        # Compare out of samples
                        #
                        col_names =  [r'Sens. [\%]', r'SP [\%]', r'Spec. [\%]']
                        colorize( col_names, op_name, color, color_map)

                        lines = []
                        lines += [ HLine(_contextManaged = False) ]
                        lines += [ HLine(_contextManaged = False) ]   
                        lines += [ TableLine( columns = [''] + col_names, _contextManaged = False ) ]
                        lines += [ HLine(_contextManaged = False) ]   

                        oos_table = self.get_out_of_sample(_best_sorts)

                        for dataset_name in oos_table.dataset_name.unique():
                            exclude_keys = ['train_tag','op_name','dataset_name','true_negative','true_positive','false_negative','false_positive']
                            t = self.describe( oos_table.loc[oos_table.dataset_name==dataset_name] , exclude_keys)
                            keys    = ['sensitivity', 'sp_index', 'specificity']
                            row     = [ '%1.2f$\pm$%1.2f'%(t[key+'_mean']*100 , t[key+'_std']*100) for key in keys]
                            colorize( row, op_name, color, color_map)
                            lines += [ TableLine( columns = [ dataset_name]  + row , _contextManaged = False ) ]

                        lines += [ HLine(_contextManaged = False) ]
                        lines += [ HLine(_contextManaged = False) ]

                        # Create all tables into the PDF Latex
                        with BeamerSlide( title = f"The out-of-sample cross val. for {train_tag} in {op_name} operation."  ):
                            with Table( caption = r'The out-of-sample mean and standard dev values for each dataset.') as table:
                                with ResizeBox( size = 1. ) as rb:
                                    with Tabular( columns = '|l|' + 'ccc|' ) as tabular:
                                        tabular = tabular
                                        for line in lines:
                                            if isinstance(line, TableLine):
                                                tabular += line
                                            else:
                                                TableLine(line, rounding = None)




                    #
                    # Compare train tags
                    #
                    col_names =  [r'Sens. [\%]', r'SP [\%]', r'Spec. [\%]', r'AUC [\%]']
                    colorize( col_names, op_name, color, color_map)

                    lines = []
                    lines += [ HLine(_contextManaged = False) ]
                    lines += [ HLine(_contextManaged = False) ]   
                    lines += [ TableLine( columns = ['', '',] + col_names, _contextManaged = False ) ]
                    lines += [ HLine(_contextManaged = False) ]   

                    for train_tag in cv_table.train_tag.unique():
                        t = cv_table.loc[ (cv_table.train_tag==train_tag) & (cv_table.op_name==op_name) ]
                        keys    = ['sens_at_test', 'sp_index_test', 'spec_at_test' , 'auc_test']
                        row     = [ '%1.2f$\pm$%1.2f'%(t[key+'_mean']*100 , t[key+'_std']*100) for key in keys]
                        colorize( row, op_name, color, color_map)
                        lines += [ TableLine( columns = [ train_tag, 'Test']  + row , _contextManaged = False ) ]

                    lines += [ HLine(_contextManaged = False) ]
                    lines += [ HLine(_contextManaged = False) ]


                    # Create all tables into the PDF Latex
                    with BeamerSlide( title = f"The in-sample cross val for each method in {op_name} operation."  ):
                        with Table( caption = r'The in-sample mean and standard dev. values for each method.') as table:
                            with ResizeBox( size = 1. ) as rb:
                                with Tabular( columns = '|lc|' + 'cccc|' ) as tabular:
                                    tabular = tabular
                                    for line in lines:
                                        if isinstance(line, TableLine):
                                            tabular += line
                                        else:
                                            TableLine(line, rounding = None)



                    #
                    # Compare out of samples train_tags
                    #
                    col_names =  [r'Sens. [\%]', r'SP [\%]', r'Spec. [\%]']
                    colorize( col_names, op_name, color, color_map)
                    lines = []
                    lines += [ HLine(_contextManaged = False) ]
                    lines += [ HLine(_contextManaged = False) ]   
                    lines += [ TableLine( columns = ['', '',] + col_names, _contextManaged = False ) ]
                    lines += [ HLine(_contextManaged = False) ]   

                    for train_tag in cv_table.train_tag.unique():
                        _best_sorts = best_sorts.loc[(best_sorts.train_tag==train_tag)&(best_sorts.op_name==op_name)]
                        oos_table = self.get_out_of_sample(_best_sorts)
                        dataset_names = oos_table.dataset_name.unique()
                        for idx, dataset_name in enumerate(dataset_names):
                            exclude_keys = ['train_tag','op_name','dataset_name','true_negative','true_positive','false_negative','false_positive']
                            t = self.describe( oos_table.loc[oos_table.dataset_name==dataset_name] , exclude_keys)
                            keys    = ['sensitivity', 'sp_index', 'specificity']
                            row     = [ '%1.2f$\pm$%1.2f'%(t[key+'_mean']*100 , t[key+'_std']*100) for key in keys]
                            colorize( row, op_name, color, color_map)
                            if idx > 0:
                                lines += [ TableLine( columns = [ '', dataset_name]  + row , _contextManaged = False ) ]
                            else:
                                multirow ='\multirow{'+str(len(dataset_names))+'}{*}{'+train_tag+'}'
                                lines += [ TableLine( columns = [ multirow, dataset_name]  + row , _contextManaged = False ) ]
                        lines += [ HLine(_contextManaged = False) ]


                    lines += [ HLine(_contextManaged = False) ]
                    
                    # Create all tables into the PDF Latex
                    with BeamerSlide( title = f"The out-of-sample cross val. for each method in {op_name} operation."  ):
                        with Table( caption = r'The out-of-sample mean and stadard dev. values for each method.') as table:
                            with ResizeBox( size = 0.8 ) as rb:
                                with Tabular( columns = '|lc|' + 'ccc|' ) as tabular:
                                    tabular = tabular
                                    for line in lines:
                                        if isinstance(line, TableLine):
                                            tabular += line
                                        else:
                                            TableLine(line, rounding = None)



if __name__ == "__main__":


    def create_op_dict( op_name, extra_suf="" ):

        d = collections.OrderedDict( {
                'max_sp'           : f'summary{extra_suf}/max_sp',
                'auc'              : f'summary{extra_suf}/auc',
                #'acc'              : f'summary{extra_suf}/acc',
                #'pd'               : f'summary{extra_suf}/pd',
                #'fa'               : f'summary{extra_suf}/fa',
                'sens'             : f'summary{extra_suf}/sensitivity',
                'spec'             : f'summary{extra_suf}/specificity',
                'threshold'        : f'summary{extra_suf}/threshold',
                'roc'              : f'summary{extra_suf}/roc',
                'roc_val'          : f'summary{extra_suf}/roc_val',
                'roc_op'           : f'summary{extra_suf}/roc_op',
                'roc_test'         : f'summary{extra_suf}/roc_test',

                'min_spec_sens_reached' : f'{op_name}{extra_suf}/min_spec_sens_reached',

                'max_sp_val'       : f'summary{extra_suf}/max_sp_val',
                'auc_val'          : f'summary{extra_suf}/auc_val',
                #'acc_val'          : f'summary{extra_suf}/acc_val',
                #'pd_val'           : f'summary{extra_suf}/pd_val',
                #'fa_val'           : f'summary{extra_suf}/fa_val',
                'sens_val'         : f'summary{extra_suf}/sensitivity_val',
                'spec_val'         : f'summary{extra_suf}/specificity_val', 

                'max_sp_test'      : f'summary{extra_suf}/max_sp_test',
                'auc_test'         : f'summary{extra_suf}/auc_test',
                #'acc_test'         : f'summary{extra_suf}/acc_test',
                #'pd_test'          : f'summary{extra_suf}/pd_test',
                #'fa_test'          : f'summary{extra_suf}/fa_test',
                'sens_test'        : f'summary{extra_suf}/sensitivity_test',
                'spec_test'        : f'summary{extra_suf}/specificity_test', 

                'max_sp_op'        : f'summary{extra_suf}/max_sp_op',
                'auc_op'           : f'summary{extra_suf}/auc_op',
                #'acc_op'           : f'summary{extra_suf}/acc_op',
                #'pd_op'            : f'summary{extra_suf}/pd_op',
                #'fa_op'            : f'summary{extra_suf}/fa_op',
                'sens_op'          : f'summary{extra_suf}/sensitivity_op',
                'spec_op'          : f'summary{extra_suf}/specificity_op', 

                'sp_index'         : f'{op_name}{extra_suf}/sp_index',
                'sens_at'          : f'{op_name}{extra_suf}/sensitivity',
                'spec_at'          : f'{op_name}{extra_suf}/specificity',
                'acc_at'           : f'{op_name}{extra_suf}/acc',
                'threshold_at'     : f'{op_name}{extra_suf}/threshold',

                'sp_index_val'     : f'{op_name}{extra_suf}/sp_index_val',
                'sens_at_val'      : f'{op_name}{extra_suf}/sensitivity_val',
                'spec_at_val'      : f'{op_name}{extra_suf}/specificity_val',
                #'acc_at_val'       : f'{op_name}{extra_suf}/acc_val',
                'threshold_at_val' : f'{op_name}{extra_suf}/threshold_val',

                'sp_index_test'    : f'{op_name}{extra_suf}/sp_index_test',
                'sens_at_test'     : f'{op_name}{extra_suf}/sensitivity_test',
                'spec_at_test'     : f'{op_name}{extra_suf}/specificity_test',
                #'acc_at_test'      : f'{op_name}{extra_suf}/acc_test',
                'threshold_at_test': f'{op_name}{extra_suf}/threshold_test',

                'sp_index_op'      : f'{op_name}{extra_suf}/sp_index_op',
                'sens_at_op'       : f'{op_name}{extra_suf}/sensitivity_op',
                'spec_at_op'       : f'{op_name}{extra_suf}/specificity_op',
                #'acc_at_op'        : f'{op_name}{extra_suf}/acc_op',
                'threshold_at_op'  : f'{op_name}{extra_suf}/threshold_op',

                'inference'        : f'{op_name}{extra_suf}/inference',
        })
        return d


    extra_suf='_val'
    #extra_suf=''
    conf_dict = collections.OrderedDict(
        {
            #'loose'   : create_op_dict( 'loose'  , extra_suf=extra_suf ),
            'medium'  : create_op_dict( 'medium' , extra_suf=extra_suf),
            #'tight'   : create_op_dict( 'tight'  , extra_suf=extra_suf),
        }
    )
    pprint(conf_dict)

    models = [
        ( 'user.philipp.gaspar.convnets.baseline.shenzhen_santacasa.exp.989f87bed5.r1'                          , 'base.sh-sc.e'        ),
        ( 'user.philipp.gaspar.convnets.altogether.shenzhen_santacasa.exp_wgan_p2p.67de4190c1.r1'               , 'alto.sh-sc.ewp'      ),
        ( 'user.philipp.gaspar.convnets.interleaved.shenzhen_santacasa.exp_wgan_p2p.e540d24b4b.r1'              , 'inte.sh-sc.ewp'      ),
        ( 'user.philipp.gaspar.convnets.altogether.shenzhen_santacasa.exp_wgan_p2p_cycle.a19a3a4f8c.r1'         , 'alto.sh-sc.ewpc'     ),
        ( 'user.philipp.gaspar.convnets.interleaved.shenzhen_santacasa.exp_wgan_p2p_cycle.a19a3a4f8c.r1'        , 'inte.sh-sc.ewpc'     ),
    ]
    models_plus_manaus = [
        # plus manaus
        ( 'user.philipp.gaspar.convnets.baseline.shenzhen_santacasa_manaus.exp.ffe6cbee11.r1'                   , 'base.sh-sc-ma.e'     ),
        ( 'user.philipp.gaspar.convnets.altogether.shenzhen_santacasa_manaus.exp_wgan_p2p.0d13030165.r1'        , 'alto.sh-sc-ma.ewp'   ),
        ( 'user.philipp.gaspar.convnets.interleaved.shenzhen_santacasa_manaus.exp_wgan_p2p.ac79954ba0.r1'       , 'inte.sh-sc-ma.ewp'   ),
        ( 'user.philipp.gaspar.convnets.altogether.shenzhen_santacasa_manaus.exp_wgan_p2p_cycle.c5143abd1b.r1'  , 'alto.sh-sc-ma.ewpc'  ),
        ( 'user.philipp.gaspar.convnets.interleaved.shenzhen_santacasa_manaus.exp_wgan_p2p_cycle.c5143abd1b.r1' , 'inte.sh-sc-ma.ewpc'  ),
    ]

    basepath='/mnt/brics_data/models'

    for path, train_tag in models+models_plus_manaus:
        cv = crossval_table( conf_dict )
        cv.fill( basepath+'/'+path , train_tag )
        table = cv.table
        best_sorts = cv.filter_sorts( table,      'sp_index_op'    )
        best_tests = cv.filter_tests( best_sorts, 'sp_index_test' )
        cv.report(table, best_sorts, best_tests, train_tag, train_tag )

    
    cv = crossval_table( conf_dict )
    for path, train_tag in models:
        cv.fill( basepath+'/'+path , train_tag )
    table      = cv.table
    best_sorts = cv.filter_sorts( table,      'sp_index_op'    )
    best_tests = cv.filter_tests( best_sorts, 'sp_index_test' )
    cv.report(table, best_sorts, best_tests, 'shenzhen_santacasa', 'shenzhen_santacasa' , detailed=False)
  

    cv = crossval_table( conf_dict )
    for path, train_tag in models_plus_manaus:
        cv.fill( basepath+'/'+path , train_tag )
    table      = cv.table
    best_sorts = cv.filter_sorts( table,      'sp_index_op'    )
    best_tests = cv.filter_tests( best_sorts, 'sp_index_test' )
    cv.report(table, best_sorts, best_tests, 'shenzhen_santacasa_manaus', 'shenzhen_santacasa_manaus' , detailed=False )
  


