

import utils.convnets as convnets
import collections
import pandas as pd
import numpy as np
from loguru import logger
from sklearn.metrics import auc, confusion_matrix, roc_curve




def evaluate( train_state, train_data, valid_data, test_data ):

    decorators = [
                    Summary( key = 'summary', detailed=True ),
                    OMS( key = "oms" , min_sensitivity = 0.9, min_specificity = 0.7, detailed=True)
                ]
    for decor in decorators:
        decor( train_state, train_data, valid_data, test_data )
    return train_state



class Summary:

    def __init__(self, key, batch_size=32, detailed=False):
        self.key = key
        self.batch_size=batch_size
        self.detailed = detailed


    def __call__( self, train_state, train_data, valid_data, test_data ):

        model, history, params = convnets.build_model_from_train_state(train_state)

        d            = collections.OrderedDict()
        op_data      = pd.concat([train_data, valid_data])

        ds_train     = convnets.build_dataset(train_data, params["image_shape"], batch_size=self.batch_size)
        ds_valid     = convnets.build_dataset(valid_data, params["image_shape"], batch_size=self.batch_size)
        ds_test      = convnets.build_dataset(test_data , params["image_shape"], batch_size=self.batch_size)
        ds_operation = convnets.build_dataset(op_data   , params["image_shape"], batch_size=self.batch_size)

        # set threshold by validation set
        metrics_train, threshold = self.calculate( ds_train    , train_data    , model  )              
        metrics_val  , _         = self.calculate( ds_valid    , valid_data    , model , label="_val" , threshold=threshold )
        metrics_test , _         = self.calculate( ds_test     , test_data     , model , label="_test", threshold=threshold ) 
        metrics_op   , _         = self.calculate( ds_operation, op_data       , model , label="_op"  , threshold=threshold )


        for metrics in [metrics_val, metrics_train, metrics_test, metrics_op]:
            d.update(metrics)


        train_state.history[self.key] = d

        logger.info( "Train     : SP = %1.2f (Sens = %1.2f, Spec = %1.2f), AUC = %1.2f" % (d['sp_max']*100     , d['sensitivity']*100     , d['specificity']*100, d['auc']))
        logger.info( "Valid     : SP = %1.2f (Sens = %1.2f, Spec = %1.2f), AUC = %1.2f" % (d['sp_max_val']*100 , d['sensitivity_val']*100 , d['specificity_val']*100, d['auc_val']))
        logger.info( "Test      : SP = %1.2f (Sens = %1.2f, Spec = %1.2f), AUC = %1.2f" % (d['sp_max_test']*100, d['sensitivity_test']*100, d['specificity_test']*100, d['auc_test']))
        logger.info( "Operation : SP = %1.2f (Sens = %1.2f, Spec = %1.2f), AUC = %1.2f" % (d['sp_max_op']*100  , d['sensitivity_op']*100  , d['specificity_op']*100, d['auc_op']))

        return train_state



    def calculate( self, ds, df, model, label='', threshold=None):

        metrics = collections.OrderedDict({})

        y_true  = df["label"].values.astype(int)
        y_prob  = model.predict(ds).squeeze()
    
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

        # given the roc
        metrics["sp_max" +label]     = sp_values[knee]
        metrics["auc"    +label]     = auc(fpr, tpr)
        metrics["roc"    +label]     = {"fpr":fpr, "tpr":tpr, "thresholds":thresholds}
        print(f'label = {label} , tn = {tn} , tp = {tp} , fn = {fn} , fp = {fp}')

        if self.detailed:
            # calculate predictions
            d = {
                "image_name" : df["path"].apply(lambda x: x.split("/")[-1].split(".")[0]),
                "y_prob"     : y_prob,
                "y_pred"     : y_pred,
            }
            metrics["predictions"] = pd.DataFrame.from_dict(d)
     
  
        return metrics, thr




class OMS:

    def __init__(self, key, batch_size=32, min_sensitivity=0.9, min_specificity=0.7, detailed=False):
        self.key = key
        self.batch_size=batch_size
        self.detailed = detailed
        self.min_sensitivity = min_sensitivity
        self.min_specificity = min_specificity


    def __call__( self, train_state, train_data, valid_data, test_data ):

        model, history, params = convnets.build_model_from_train_state(train_state)

        d            = collections.OrderedDict()
        op_data      = pd.concat([train_data, valid_data])

        ds_train     = convnets.build_dataset(train_data, params["image_shape"], batch_size=self.batch_size)
        ds_valid     = convnets.build_dataset(valid_data, params["image_shape"], batch_size=self.batch_size)
        ds_test      = convnets.build_dataset(test_data , params["image_shape"], batch_size=self.batch_size)
        ds_operation = convnets.build_dataset(op_data   , params["image_shape"], batch_size=self.batch_size)

        # set threshold by validation set
        metrics_train, threshold = self.calculate( ds_train    , train_data    , model  )              

        if threshold: # if threshold is not none, we have a cut higher than OMS roc curve area
            metrics_val  , _         = self.calculate( ds_valid    , valid_data    , model , label="_val" , threshold=threshold )
            metrics_test , _         = self.calculate( ds_test     , test_data     , model , label="_test", threshold=threshold ) 
            metrics_op   , _         = self.calculate( ds_operation, op_data       , model , label="_op"  , threshold=threshold )

            # update everything
            for metrics in [metrics_val, metrics_train, metrics_test, metrics_op]:
                d.update(metrics)


        train_state.history[self.key] = d # if not OMS area, this d will be empty ({})

        if d:
            logger.info( "Train     : SP = %1.2f (Sens = %1.2f, Spec = %1.2f), AUC = %1.2f" % (d['sp_max']*100     , d['sensitivity']*100     , d['specificity']*100, d['auc']))
            logger.info( "Valid     : SP = %1.2f (Sens = %1.2f, Spec = %1.2f), AUC = %1.2f" % (d['sp_max_val']*100 , d['sensitivity_val']*100 , d['specificity_val']*100, d['auc_val']))
            logger.info( "Test      : SP = %1.2f (Sens = %1.2f, Spec = %1.2f), AUC = %1.2f" % (d['sp_max_test']*100, d['sensitivity_test']*100, d['specificity_test']*100, d['auc_test']))
            logger.info( "Operation : SP = %1.2f (Sens = %1.2f, Spec = %1.2f), AUC = %1.2f" % (d['sp_max_op']*100  , d['sensitivity_op']*100  , d['specificity_op']*100, d['auc_op']))
        else:
            logger.info("Not inside of OMS roc curve area...")

        return train_state



    def calculate( self, ds, df, model, label='', threshold=None):

        metrics = collections.OrderedDict({})

        y_true = df["label"].values.astype(int)
        y_prob = model.predict(ds).squeeze()
    
        # get roc curve
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        sp_values = np.sqrt(np.sqrt(tpr * (1 - fpr)) * (0.5 * (tpr + (1 - fpr))))
        knee      = np.argmax(sp_values)
        sp_max    = sp_values[knee]
        thr       = None

        if not threshold:
            # calculate metrics inside the WHO area
            who_selection = (tpr >= self.min_sensitivity) & ((1 - fpr) >= self.min_specificity)

            if np.any(who_selection):
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

            print(f'label = {label} , tn = {tn} , tp = {tp} , fn = {fn} , fp = {fp}')
            # given the roc
            metrics["sp_max" +label]     = sp_max
            metrics["auc"    +label]     = auc(fpr, tpr)
            metrics["roc"    +label]     = {"fpr":fpr, "tpr":tpr, "thresholds":thresholds}


            if self.detailed:
                # calculate predictions
                d = {
                    "image_name" : df["path"].apply(lambda x: x.split("/")[-1].split(".")[0]),
                    "y_prob"     : y_prob,
                    "y_pred"     : y_pred,
                }
                metrics["predictions"] = pd.DataFrame.from_dict(d)
  
        return metrics, thr
