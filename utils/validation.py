

import utils.convnets

from loguru import logger

class Summary:

    def __init__(self, key, batch_size=128, mim_sensibility=0.9, min_specificity=0.7, detailed=False):
        self.key = key
        self.batch_size=batch_size
        self.detailed = detailed
        self.min_sensitivity = min_sensitivity
        self.min_specificity = min_specificity


    def __call__( self, train_state, train_data, valid_data, test_data ):

        model, history, params = convnets.build_model_from_train_state(train_state)

        d = collections.OrderedDict()
        ds_train     = convnets.build_dataset(train_data, params["image_shape"], batch_size=self.batch_size)
        ds_valid     = convnets.build_dataset(valid_data, params["image_shape"], batch_size=self.batch_size)
        ds_test      = convnets.build_dataset(test_data , params["image_shape"], batch_size=self.batch_size)
        ds_operation = convnets.build_dataset(pd.concat([train_data, df_valid_data]), params["image_shape"], batch_size=self.batch_size)

        self.calculate( d, ds_train    , train_data    , model                )
        self.calculate( d, ds_valid    , valid_data    , model , label="_val" )
        self.calculate( d, ds_test     , test_data     , model , label="_test")
        self.calculate( d, ds_operation, operation_data, model , label="_op"  )

        train_state.history[self.key] = d


        logger.info( "Train     : SP = %1.2f (Sens = %1.2f, Spec = %1.2f), AUC = %1.2f" % (d['sp_max']*100     , d['sensitivity']*100     , d['specificity']*100, d['auc']))
        logger.info( "Valid     : SP = %1.2f (Sens = %1.2f, Spec = %1.2f), AUC = %1.2f" % (d['sp_max_val']*100 , d['sensitivity_val']*100 , d['specificity_val']*100, d['auc_val']))
        logger.info( "Test      : SP = %1.2f (Sens = %1.2f, Spec = %1.2f), AUC = %1.2f" % (d['sp_max_test']*100, d['sensitivity_test']*100, d['specificity_test']*100, d['auc']))
        logger.info( "Operation : SP = %1.2f (Sens = %1.2f, Spec = %1.2f), AUC = %1.2f" % (d['sp_max_op']*100  , d['sensitivity_op']*100  , d['specificity_test']*100, d['auc']))

        return train_state



    def calculate( metrics , ds, df, model, label=''):

        y_true = df["label"].values.astype(int)
        y_prob = model.predict(ds).squeeze()
    
        # get roc curve
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
   
        # calculate the total SP & AUC
        sp_values = np.sqrt(np.sqrt(tpr * (1 - fpr)) * (0.5 * (tpr + (1 - fpr))))
        knee      = np.argmax(sp_values)
        threshold = thresholds[knee]
        y_pred    = (y_prob >= threshold).astype(int)

        # confusion matrix and metrics
        conf_matrix                  = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp               = conf_matrix.ravel()
        metrics["sensitivity"+label] = tp / (tp + fn)
        metrics["specificity"+label] = tn / (tn + fp)
        metrics["precision"+label]   = tp / (tp + fp)
        metrics["recall"+label]      = tp / (tp + fn)
        metrics["sp_max" +label]     = sp_values[knee]
        metrics["auc"    +label]     = auc(fpr, tpr)
        metrics['threshold'+label]   = threshold
          
    return metrics




class Summary:

    def __init__(self, key, batch_size=128, mim_sensibility=0.9, min_specificity=0.7, detailed=False):
        self.key = key
        self.batch_size=batch_size
        self.detailed = detailed
        self.min_sensitivity = min_sensitivity
        self.min_specificity = min_specificity


    def __call__( self, train_state, train_data, valid_data, test_data ):

        model, history, params = convnets.build_model_from_train_state(train_state)

        d = collections.OrderedDict()
        ds_train     = convnets.build_dataset(train_data, params["image_shape"], batch_size=self.batch_size)
        ds_valid     = convnets.build_dataset(valid_data, params["image_shape"], batch_size=self.batch_size)
        ds_test      = convnets.build_dataset(test_data , params["image_shape"], batch_size=self.batch_size)
        ds_operation = convnets.build_dataset(pd.concat([train_data, df_valid_data]), params["image_shape"], batch_size=self.batch_size)

        self.calculate( d, ds_train    , train_data    , model                )
        self.calculate( d, ds_valid    , valid_data    , model , label="_val" )
        self.calculate( d, ds_test     , test_data     , model , label="_test")
        self.calculate( d, ds_operation, operation_data, model , label="_op"  )

        train_state.history[self.key] = d


        logger.info( "Train     : SP = %1.2f (Sens = %1.2f, Spec = %1.2f), AUC = %1.2f (WHO = %s)" % (d['sp_max']*100, d[]))

        return train_state



    def calculate( metrics , ds, df, model, label=''):


        y_true = df["label"].values.astype(int)
        y_prob = model.predict(ds).squeeze()
    
        # get roc curve
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
   
        # calculate the total SP & AUC
        sp_values = np.sqrt(np.sqrt(tpr * (1 - fpr)) * (0.5 * (tpr + (1 - fpr))))
        knee      = np.argmax(sp_values)
        threshold = thresholds[knee]
        y_pred    = (y_prob >= threshold).astype(int)

        # confusion matrix and metrics
        conf_matrix                  = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp               = conf_matrix.ravel()
        metrics["sensitivity"+label] = tp / (tp + fn)
        metrics["specificity"+label] = tn / (tn + fp)
        metrics["precision"+label]   = tp / (tp + fp)
        metrics["recall"+label]      = tp / (tp + fn)
        metrics["sp_max" +label]     = sp_values[knee]
        metrics["auc"    +label]     = auc(fpr, tpr)

        
        # check WHO


        # calculate metrics inside the WHO area
        who_selection = (tpr >= self.min_sensitivity) & ((1 - fpr) >= self.min_specificity)
        
        # select the best threshold
        if np.any(who_selection):
            # get the thrshold after the WHO area by sp 
            metrics["who"+label] = True
            sp_values = sp_values[who_selection]
            sp_argmax = np.argmax(sp_values)
            metrics["sp_max_who"+label] = sp_values[sp_argmax]
            if sum(who_selection) <= 1:
                metrics["auc_who"+label] = None
            else:
                metrics["auc_who"+label] = auc(fpr[who_selection], tpr[who_selection])
            metrics["threshold"+label] = thresholds[who_selection][sp_argmax]
        else:
            # get the threshold from the knee of roc curve
            metrics["who"+label]        = False
            metrics["sp_max_who"+label] = 0.0
            metrics["auc_who"+label]    = 0.0
            metrics["threshold"+label]  = thresholds[knee]


        # make predictions using the WHO threshold
        y_pred = (y_prob >= metrics["threshold"+label]).astype(int)
        y_true = df["label"].values.astype(int)
    
        if self.detailed:
            # calculate predictions
            aux_metrics["image_name"] = df["path"].apply(
                lambda x: x.split("/")[-1].split(".")[0]
            )
            aux_metrics["y_prob"] = y_prob
            aux_metrics["y_pred"] = y_pred
            metrics["predictions"] = pd.DataFrame.from_dict(aux_metrics)
            metrics["fpr"+label] = fpr
            metrics["tpr"+label] = tpr
    
    
        # confusion matrix and metrics
        conf_matrix = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = conf_matrix.ravel()
        metrics["sensitivity"+label] = tp / (tp + fn)
        metrics["specificity"+label] = tn / (tn + fp)
        metrics["precision"+label] = tp / (tp + fp)
        metrics["recall"+label] = tp / (tp + fn)
        
        
        
    
  
    return metrics
