__all__ = ["evaluate"]

from pprint import pprint
from itertools import product
from tqdm import tqdm
from loguru import logger
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.metrics import mean_squared_error

import collections, os, glob, json, copy, re, pickle
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import screening.utils.convnets as convnets




def evaluate( train_state, train_data, valid_data, test_data ):

    decorators = [
                    Summary( key = 'summary' ),
                    Reference( 'loose'   , sensitivity=0.9  ), # 0.9 >= of detection
                    Reference( 'medium'  , sensitivity=0.9, specificity=0.7  ), # pd >= 0.9 and fa =< 0.3, best sp inside of this region
                    Reference( 'tight'   , specificity=0.7  ), # 0.3 <= of fake
                ]

    cache = {}
    for decor in decorators:
        decor( train_state, train_data, valid_data, test_data, cache=cache )
    return train_state



#
# train summary given best sp thrshold selection
#
class Summary:

    def __init__(self, key, batch_size=32):
        self.key = key
        self.batch_size=batch_size


    def __call__( self, train_state, train_data, valid_data, test_data, cache={} ):

        model, history, params = convnets.build_model_from_train_state(train_state)

        d            = collections.OrderedDict()
        op_data      = pd.concat([train_data, valid_data])

        ds_train     = convnets.build_dataset(train_data, params["image_shape"], batch_size=self.batch_size)
        ds_valid     = convnets.build_dataset(valid_data, params["image_shape"], batch_size=self.batch_size)
        ds_test      = convnets.build_dataset(test_data , params["image_shape"], batch_size=self.batch_size)
        ds_operation = convnets.build_dataset(op_data   , params["image_shape"], batch_size=self.batch_size)

        # set threshold by validation set
        metrics_train, threshold = self.calculate( ds_train    , train_data    , model , cache )              
        metrics_val  , _         = self.calculate( ds_valid    , valid_data    , model , cache, label="_val" , threshold=threshold )
        metrics_test , _         = self.calculate( ds_test     , test_data     , model , cache, label="_test", threshold=threshold ) 
        metrics_op   , _         = self.calculate( ds_operation, op_data       , model , cache, label="_op"  , threshold=threshold )


        for metrics in [metrics_val, metrics_train, metrics_test, metrics_op]:
            d.update(metrics)


        train_state.history[self.key] = d

        logger.info( "Train     : SP = %1.2f (Sens = %1.2f, Spec = %1.2f), AUC = %1.2f" % (d['max_sp']*100     , d['sensitivity']*100     , d['specificity']*100, d['auc']))
        logger.info( "Valid     : SP = %1.2f (Sens = %1.2f, Spec = %1.2f), AUC = %1.2f" % (d['max_sp_val']*100 , d['sensitivity_val']*100 , d['specificity_val']*100, d['auc_val']))
        logger.info( "Test      : SP = %1.2f (Sens = %1.2f, Spec = %1.2f), AUC = %1.2f" % (d['max_sp_test']*100, d['sensitivity_test']*100, d['specificity_test']*100, d['auc_test']))
        logger.info( "Operation : SP = %1.2f (Sens = %1.2f, Spec = %1.2f), AUC = %1.2f" % (d['max_sp_op']*100  , d['sensitivity_op']*100  , d['specificity_op']*100, d['auc_op']))

        return train_state



    def calculate( self, ds, df, model, cache, label='', threshold=None):

        metrics = collections.OrderedDict({})

        y_true  = df["label"].values.astype(int)

        if f'y_prob{label}' not in cache.keys() :
            y_prob  = model.predict(ds).squeeze() 
            cache[f'y_prob{label}'] = y_prob
        else:
            y_prob = cache[f'y_prob{label}']
    
        # get roc curve
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        # calculate the total SP & AUC
        sp_values = np.sqrt(np.sqrt(tpr * (1 - fpr)) * (0.5 * (tpr + (1 - fpr))))
        knee      = np.argmax(sp_values)
        thr       = thresholds[knee] if not threshold else threshold
        y_pred    = (y_prob >= thr).astype(int)

        # confusion matrix and metrics
        conf_matrix                  = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp               = conf_matrix.ravel()
        fa = fp / (tn + fp)
        det = tp / (tp + fn) # same as recall or sensibility

        # given the threshold
        metrics['threshold'+label]      = thr
        metrics["sp_index"+label]       = np.sqrt(np.sqrt(det * (1 - fa)) * (0.5 * (det + (1 - fa))))
        metrics["fa"+label]             = fa
        metrics["pd"+label]             = det
        metrics["sensitivity"+label]    = tp / (tp + fn) if (tp+fn) > 0 else 0 # same as recall
        metrics["specificity"+label]    = tn / (tn + fp) if (tn+fp) > 0 else 0
        metrics["precision"+label]      = tp / (tp + fp) if (tp+fp) > 0 else 0
        metrics["recall"+label]         = tp / (tp + fn) if (tp+fn) > 0 else 0# same as sensibility
        metrics["acc"+label]            = (tp+tn)/(tp+tn+fp+fn) # accuracy

        metrics["true_negative"+label]  = tn
        metrics["true_positive"+label]  = tp
        metrics["false_negative"+label] = fn
        metrics["false_positive"+label] = fp

        # control values
        metrics["max_sp" +label]     = sp_values[knee]
        metrics["auc"    +label]     = auc(fpr, tpr)
        metrics["roc"    +label]     = {"fpr":fpr, "tpr":tpr, "thresholds":thresholds}
        metrics["mse"    +label]     = mean_squared_error(y_true, y_prob)

        return metrics, thr




#
# 
#
class Reference:

    def __init__(self, key, batch_size=32, sensitivity=None, specificity=None):
        self.key = key
        self.batch_size=batch_size
        self.sensitivity = sensitivity
        self.specificity = specificity


    def __call__( self, train_state, train_data, valid_data, test_data, cache={} ):

        model, history, params = convnets.build_model_from_train_state(train_state)

        d            = collections.OrderedDict()
        op_data      = pd.concat([train_data, valid_data])

        ds_train     = convnets.build_dataset(train_data, params["image_shape"], batch_size=self.batch_size)
        ds_valid     = convnets.build_dataset(valid_data, params["image_shape"], batch_size=self.batch_size)
        ds_test      = convnets.build_dataset(test_data , params["image_shape"], batch_size=self.batch_size)
        ds_operation = convnets.build_dataset(op_data   , params["image_shape"], batch_size=self.batch_size)

        # set threshold by validation set
        metrics_train, threshold = self.calculate( ds_train    , train_data    , model , cache )              
        metrics_val  , _         = self.calculate( ds_valid    , valid_data    , model , cache, label="_val" , threshold=threshold )
        metrics_test , _         = self.calculate( ds_test     , test_data     , model , cache, label="_test", threshold=threshold ) 
        metrics_op   , _         = self.calculate( ds_operation, op_data       , model , cache, label="_op"  , threshold=threshold )

        # update everything
        for metrics in [metrics_val, metrics_train, metrics_test, metrics_op]:
            d.update(metrics)


        train_state.history[self.key] = d # if not OMS area, this d will be empty ({})


        if threshold:
            logger.info( "Train     : SP = %1.2f (Sens = %1.2f, Spec = %1.2f)" % (d['sp_index']*100     , d['sensitivity']*100     , d['specificity']*100     )   )
            logger.info( "Valid     : SP = %1.2f (Sens = %1.2f, Spec = %1.2f)" % (d['sp_index_val']*100 , d['sensitivity_val']*100 , d['specificity_val']*100 )   )
            logger.info( "Test      : SP = %1.2f (Sens = %1.2f, Spec = %1.2f)" % (d['sp_index_test']*100, d['sensitivity_test']*100, d['specificity_test']*100)   )
            logger.info( "Operation : SP = %1.2f (Sens = %1.2f, Spec = %1.2f)" % (d['sp_index_op']*100  , d['sensitivity_op']*100  , d['specificity_op']*100  )   )
        else:
            logger.info("Not inside of OMS roc curve area...")

        return train_state



    def calculate( self, ds, df, model, cache, label='', threshold=None):

        metrics = collections.OrderedDict({})

        y_true = df["label"].values.astype(int)

        if f'y_prob{label}' not in cache.keys() :
            y_prob  = model.predict(ds).squeeze() 
            cache[f'y_prob{label}'] = y_prob
        else:
            y_prob = cache[f'y_prob{label}']

        # get roc curve
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        sp_values = np.sqrt(np.sqrt(tpr * (1 - fpr)) * (0.5 * (tpr + (1 - fpr))))
        knee      = np.argmax(sp_values)
        sp_max    = sp_values[knee]
        thr       = None

        if not threshold:

            def closest_after( values , ref ):
              values_d = values-ref; values_d[values_d<0]=999; index = values_d.argmin()
              return values[index], index

            if self.sensitivity and not self.specificity:
                sensitivity , index = closest_after( tpr, self.sensitivity )
                thr = thresholds[index]
                logger.info("closest sensitivity as %1.2f (%1.2f)" % (sensitivity, self.sensitivity))

            elif self.specificity and not self.sensitivity:
                specificity , index = closest_after( 1-fpr, self.specificity )
                thr = thresholds[index]
                logger.info("closest specificity as %1.2f (%1.2f)" % (specificity, self.specificity))

            else: # sp max inside of the area

                # calculate metrics inside the WHO area
                who_selection = (tpr >= self.sensitivity) & ((1 - fpr) >= self.specificity)
                if np.any(who_selection):
                    logger.info("selection inside of the WHO area")
                    sp_values = sp_values[who_selection]
                    sp_argmax = np.argmax(sp_values)
                    thr       = thresholds[sp_argmax]
        else:
            thr = threshold



        if thr:

            y_pred    = (y_prob >= thr).astype(int)
            # confusion matrix and metrics
            conf_matrix                  = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp               = conf_matrix.ravel()
            fa = fp / (tn + fp)
            det = tp / (tp + fn) # same as recall or sensibility

            # given the threshold
            metrics['threshold'+label]      = thr
            metrics["sp_index"+label]       = np.sqrt(np.sqrt(det * (1 - fa)) * (0.5 * (det + (1 - fa))))
            metrics["pd"+label]             = det
            metrics['fa'+label]             = fa
            metrics["sensitivity"+label]    = tp / (tp + fn) if (tp+fn) > 0 else 0 # same as recall
            metrics["specificity"+label]    = tn / (tn + fp) if (tn+fp) > 0 else 0
            metrics["precision"+label]      = tp / (tp + fp) if (tp+fp) > 0 else 0
            metrics["recall"+label]         = tp / (tp + fn) if (tp+fn) > 0 else 0# same as sensibility
            metrics["acc"+label]            = (tp+tn)/(tp+tn+fp+fn) # accuracy
            metrics["true_negative"+label]  = tn
            metrics["true_positive"+label]  = tp
            metrics["false_negative"+label] = fn
            metrics["false_positive"+label] = fp

        else:
            # given the threshold
            metrics['threshold'+label]      = -1
            metrics["sp_index"+label]       = -1
            metrics["pd"+label]             = -1
            metrics["fa"+label]             = -1
            metrics["sensitivity"+label]    = -1 # same as recall
            metrics["specificity"+label]    = -1
            metrics["precision"+label]      = -1
            metrics["recall"+label]         = -1 # same as sensibility
            metrics["acc"+label]            = -1
            metrics["true_negative"+label]  = -1
            metrics["true_positive"+label]  = -1
            metrics["false_negative"+label] = -1
            metrics["false_positive"+label] = -1


        return metrics, thr








