

import utils.convnets

from loguru import logger



def evaluate( train_state, data_train, data_valid, data_test ):

    decorators = [
                    Summary( key = 'summary', detailed=True ),
                    OWS( key = "who" , min_sensitivity = , min_specificity = , detailed=True)
                ]
    for decor in decorators:
        decor( train_state, train_data, valid_data, test_data )
    return train_state



class Summary:

    def __init__(self, key, batch_size=128, detailed=False):
        self.key = key
        self.batch_size=batch_size
        self.detailed = detailed


    def __call__( self, train_state, train_data, valid_data, test_data ):

        model, history, params = convnets.build_model_from_train_state(train_state)

        d = collections.OrderedDict()

        ds_train     = convnets.build_dataset(train_data, params["image_shape"], batch_size=self.batch_size)
        ds_valid     = convnets.build_dataset(valid_data, params["image_shape"], batch_size=self.batch_size)
        ds_test      = convnets.build_dataset(test_data , params["image_shape"], batch_size=self.batch_size)
        ds_operation = convnets.build_dataset(pd.concat([train_data, df_valid_data]), params["image_shape"], batch_size=self.batch_size)


        d.update( self.calculate( ds_train    , train_data    , model                ) )
        d.update( self.calculate( ds_valid    , valid_data    , model , label="_val" ) )
        d.update( self.calculate( ds_test     , test_data     , model , label="_test") )
        d.update( self.calculate( ds_operation, operation_data, model , label="_op"  ) )

        train_state.history[self.key] = d

        logger.info( "Train     : SP = %1.2f (Sens = %1.2f, Spec = %1.2f), AUC = %1.2f" % (d['sp_max']*100     , d['sensitivity']*100     , d['specificity']*100, d['auc']))
        logger.info( "Valid     : SP = %1.2f (Sens = %1.2f, Spec = %1.2f), AUC = %1.2f" % (d['sp_max_val']*100 , d['sensitivity_val']*100 , d['specificity_val']*100, d['auc_val']))
        logger.info( "Test      : SP = %1.2f (Sens = %1.2f, Spec = %1.2f), AUC = %1.2f" % (d['sp_max_test']*100, d['sensitivity_test']*100, d['specificity_test']*100, d['auc']))
        logger.info( "Operation : SP = %1.2f (Sens = %1.2f, Spec = %1.2f), AUC = %1.2f" % (d['sp_max_op']*100  , d['sensitivity_op']*100  , d['specificity_test']*100, d['auc']))

        return train_state



    def calculate( ds, df, model, label=''):

        metrics = collections.OrderedDict({})

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




class OWS:

    def __init__(self, key, batch_size=128, mim_sensibility=0.9, min_specificity=0.7, detailed=False):
        self.key = key
        self.batch_size=batch_size
        self.detailed = detailed
        self.min_sensitivity = min_sensitivity
        self.min_specificity = min_specificity


    def __call__( self, train_state, train_data, valid_data, test_data ):

        model, history, params = convnets.build_model_from_train_state(train_state)

        d            = collections.OrderedDict()
        ds_train     = convnets.build_dataset(train_data, params["image_shape"], batch_size=self.batch_size)
        ds_valid     = convnets.build_dataset(valid_data, params["image_shape"], batch_size=self.batch_size)
        ds_test      = convnets.build_dataset(test_data , params["image_shape"], batch_size=self.batch_size)
        ds_operation = convnets.build_dataset(pd.concat([train_data, df_valid_data]), params["image_shape"], batch_size=self.batch_size)

        d.update( self.calculate( ds_train    , train_data    , model                ) )
        d.update( self.calculate( ds_valid    , valid_data    , model , label="_val" ) )
        d.update( self.calculate( ds_test     , test_data     , model , label="_test") )
        d.update( self.calculate( ds_operation, operation_data, model , label="_op"  ) )

        train_state.history[self.key] = d


        if d['who_area']:
            logger.info( "Training inside OWS area...")
            logger.info( "Train     (OWS) : SP = %1.2f (Sens = %1.2f, Spec = %1.2f), AUC = %1.2f" % (d['sp_max']*100     , d['sensitivity']*100     , d['specificity']*100, d['auc']))
            logger.info( "Valid     (OWS) : SP = %1.2f (Sens = %1.2f, Spec = %1.2f), AUC = %1.2f" % (d['sp_max_val']*100 , d['sensitivity_val']*100 , d['specificity_val']*100, d['auc_val']))
            logger.info( "Test      (OWS) : SP = %1.2f (Sens = %1.2f, Spec = %1.2f), AUC = %1.2f" % (d['sp_max_test']*100, d['sensitivity_test']*100, d['specificity_test']*100, d['auc']))
            logger.info( "Operation (OWS) : SP = %1.2f (Sens = %1.2f, Spec = %1.2f), AUC = %1.2f" % (d['sp_max_op']*100  , d['sensitivity_op']*100  , d['specificity_test']*100, d['auc']))
        
        return train_state



    def calculate( metrics , ds, df, model, label=''):

        metrics = collections.OrderedDict({})

        y_true = df["label"].values.astype(int)
        y_prob = model.predict(ds).squeeze()
    
        # get roc curve
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        sp_values = np.sqrt(np.sqrt(tpr * (1 - fpr)) * (0.5 * (tpr + (1 - fpr))))

        # calculate metrics inside the WHO area
        who_selection = (tpr >= self.min_sensitivity) & ((1 - fpr) >= self.min_specificity)

        metrics['who_area'] = False

        if np.any(who_selection):

            metrics['who_area'] = True
            sp_values = sp_values[who_selection]
            sp_argmax = np.argmax(sp_values)
            threshold = thresholds[sp_argmax]
            y_pred    = (y_prob >= threshold).astype(int)

            # confusion matrix and metrics
            conf_matrix                  = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp               = conf_matrix.ravel()
            metrics["sensitivity"+label] = tp / (tp + fn)
            metrics["specificity"+label] = tn / (tn + fp)
            metrics["precision"+label]   = tp / (tp + fp)
            metrics["recall"+label]      = tp / (tp + fn)
            metrics["sp_max" +label]     = sp_values[sp_argmax]
            metrics["auc"    +label]     = auc(fpr, tpr)
            metrics['threshold'+label]   = threshold
       

    
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
  
        return metrics
