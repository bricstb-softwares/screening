__all__ = ["evaluate"]

from loguru import logger
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.metrics import mean_squared_error

import collections, pickle
import numpy as np
import pandas as pd
import screening.utils.convnets as convnets
from screening import DATA_DIR





def evaluate( train_state, train_data, valid_data, test_data ):

    out_of_sample = Inference( 'inference' , ['russia', 'caxias', 'indonesia','fiocruz'] )

    decorators = [
                    # NOTE: set everythong by train dataset (right way)
                    Summary( key = 'summary', out_of_sample=out_of_sample ),
                    Reference( 'loose'      , out_of_sample=out_of_sample, sensitivity=0.9  ), # 0.9 >= of detection
                    Reference( 'medium'     , out_of_sample=out_of_sample, sensitivity=0.9, specificity=0.7  ), # pd >= 0.9 and fa =< 0.3, best sp inside of this region
                    Reference( 'tight'      , out_of_sample=out_of_sample, specificity=0.7  ), # 0.3 <= of fake

                    # NOTE: set everything by val dataset (just for )
                    Summary( key = 'summary_val', set_by_valid=True, out_of_sample=out_of_sample ),
                    Reference( 'loose_val'      , set_by_valid=True, out_of_sample=out_of_sample, sensitivity=0.9  ), # 0.9 >= of detection
                    Reference( 'medium_val'     , set_by_valid=True, out_of_sample=out_of_sample, sensitivity=0.9, specificity=0.7  ), # pd >= 0.9 and fa =< 0.3, best sp inside of this region
                    Reference( 'tight_val'      , set_by_valid=True, out_of_sample=out_of_sample, specificity=0.7  ), # 0.3 <= of fake

                ]

    # create the context
    ctx = {
            'train_state': train_state,
            'train_data' : train_data, 
            'valid_data' : valid_data, 
            'test_data'  : test_data,
            'cache'      : {},
    }
    
    d = collections.OrderedDict({})

    # decorate
    for decor in decorators:
        decor( ctx, d )

    # attach into the history
    train_state.history.update(d)

    return train_state



#
# train summary given best sp thrshold selection
#
class Summary:

    def __init__(self, key     : str, 
                 batch_size    : int=32, 
                 out_of_sample : bool=None, 
                 set_by_valid  : bool=False):

        self.key = key
        self.batch_size=batch_size
        self.out_of_sample=out_of_sample
        self.set_by_valid=set_by_valid


    def __call__( self, ctx , output_d):

        train_state  = ctx['train_state']
        train_data   = ctx['train_data']
        valid_data   = ctx['valid_data']
        test_data    = ctx['test_data' ]
        cache        = ctx['cache']

        d            = collections.OrderedDict({})

        model, history, params = convnets.build_model_from_train_state(train_state)

        op_data      = pd.concat([train_data, valid_data])
        ds_train     = convnets.build_dataset(train_data, params["image_shape"], batch_size=self.batch_size)
        ds_valid     = convnets.build_dataset(valid_data, params["image_shape"], batch_size=self.batch_size)
        ds_test      = convnets.build_dataset(test_data , params["image_shape"], batch_size=self.batch_size)
        ds_operation = convnets.build_dataset(op_data   , params["image_shape"], batch_size=self.batch_size)

        # set threshold by validation set
        if self.set_by_valid:
            metrics_val  , threshold = self.calculate( ds_valid    , valid_data    , model , cache, label="_val"  )
            metrics_train, _         = self.calculate( ds_train    , train_data    , model , cache, threshold=threshold  )                     
        else:
            metrics_train, threshold = self.calculate( ds_train    , train_data    , model , cache )              
            metrics_val  , _         = self.calculate( ds_valid    , valid_data    , model , cache, label="_val" , threshold=threshold )

        metrics_test , _         = self.calculate( ds_test     , test_data     , model , cache, label="_test", threshold=threshold ) 
        metrics_op   , _         = self.calculate( ds_operation, op_data       , model , cache, label="_op"  , threshold=threshold )


        for metrics in [metrics_val, metrics_train, metrics_test, metrics_op]:
            d.update(metrics)

        logger.info( "Train     : SP = %1.2f (Sens = %1.2f, Spec = %1.2f), AUC = %1.2f" % (d['max_sp']*100     , d['sensitivity']*100     , d['specificity']*100, d['auc']))
        logger.info( "Valid     : SP = %1.2f (Sens = %1.2f, Spec = %1.2f), AUC = %1.2f" % (d['max_sp_val']*100 , d['sensitivity_val']*100 , d['specificity_val']*100, d['auc_val']))
        logger.info( "Test      : SP = %1.2f (Sens = %1.2f, Spec = %1.2f), AUC = %1.2f" % (d['max_sp_test']*100, d['sensitivity_test']*100, d['specificity_test']*100, d['auc_test']))
        logger.info( "Operation : SP = %1.2f (Sens = %1.2f, Spec = %1.2f), AUC = %1.2f" % (d['max_sp_op']*100  , d['sensitivity_op']*100  , d['specificity_op']*100, d['auc_op']))

        if self.out_of_sample:
            ctx['threshold'] = threshold
            self.out_of_sample( ctx, d )

        output_d[self.key] = d


    #
    # calculate metrics
    #
    def calculate( self, ds, df, model, cache, label='', threshold=None):

        metrics = collections.OrderedDict({})

        y_true  = df["label"].values.astype(int)

        if f'y_prob{label}' not in cache.keys() :
            y_prob  = model.predict_generator(ds).squeeze() 
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
        conf_matrix                  = confusion_matrix(y_true, y_pred , labels=[False,True])
        tn, fp, fn, tp               = conf_matrix.ravel()
        fa = (fp / (tn + fp)) if (tn+fp) > 0 else 0
        det = (tp / (tp + fn)) if (tp+fn) > 0 else 0 # same as recall or sensibility

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

    def __init__(self, key : str, 
                 batch_size         : int=32, 
                 sensitivity        =None, 
                 specificity        =None, 
                 min_sensitivity    :float=0.9, 
                 min_specificity    : float=0.7, 
                 out_of_sample      =None,
                 set_by_valid       : bool=False):

        self.key = key
        self.batch_size=batch_size
        self.sensitivity = sensitivity
        self.specificity = specificity
        self.min_sensitivity = min_sensitivity
        self.min_specificity = min_specificity
        self.out_of_sample=out_of_sample
        self.set_by_valid=set_by_valid


    def __call__( self, ctx, output_d ):

        logger.info(f"Running reference at {self.key}")

        train_state  = ctx['train_state']
        train_data   = ctx['train_data']
        valid_data   = ctx['valid_data']
        test_data    = ctx['test_data' ]
        cache        = ctx['cache']

        model, history, params = convnets.build_model_from_train_state(train_state)

        d            = collections.OrderedDict()
        op_data      = pd.concat([train_data, valid_data])

        ds_train     = convnets.build_dataset(train_data, params["image_shape"], batch_size=self.batch_size)
        ds_valid     = convnets.build_dataset(valid_data, params["image_shape"], batch_size=self.batch_size)
        ds_test      = convnets.build_dataset(test_data , params["image_shape"], batch_size=self.batch_size)
        ds_operation = convnets.build_dataset(op_data   , params["image_shape"], batch_size=self.batch_size)

        # set threshold by validation set
        if self.set_by_valid:
            metrics_val  , threshold = self.calculate( ds_valid    , valid_data    , model , cache, label="_val"  )
            metrics_train, _         = self.calculate( ds_train    , train_data    , model , cache, threshold=threshold  )                     
        else:
            metrics_train, threshold = self.calculate( ds_train    , train_data    , model , cache )              
            metrics_val  , _         = self.calculate( ds_valid    , valid_data    , model , cache, label="_val" , threshold=threshold )

        metrics_test , _         = self.calculate( ds_test     , test_data     , model , cache, label="_test", threshold=threshold ) 
        metrics_op   , _         = self.calculate( ds_operation, op_data       , model , cache, label="_op"  , threshold=threshold )



        # update everything
        for metrics in [metrics_val, metrics_train, metrics_test, metrics_op]:
            d.update(metrics)

        if threshold:
            logger.info( "Train     : SP = %1.2f (Sens = %1.2f, Spec = %1.2f)" % (d['sp_index']*100     , d['sensitivity']*100     , d['specificity']*100     )   )
            logger.info( "Valid     : SP = %1.2f (Sens = %1.2f, Spec = %1.2f)" % (d['sp_index_val']*100 , d['sensitivity_val']*100 , d['specificity_val']*100 )   )
            logger.info( "Test      : SP = %1.2f (Sens = %1.2f, Spec = %1.2f)" % (d['sp_index_test']*100, d['sensitivity_test']*100, d['specificity_test']*100)   )
            logger.info( "Operation : SP = %1.2f (Sens = %1.2f, Spec = %1.2f)" % (d['sp_index_op']*100  , d['sensitivity_op']*100  , d['specificity_op']*100  )   )
        else:
            logger.info("Not inside of OMS roc curve area...")

        if self.out_of_sample:
            ctx['threshold'] = threshold
            self.out_of_sample( ctx, d )

        output_d[self.key] = d



    #
    # calculate metrics
    #
    def calculate( self, ds, df, model, cache, label='', threshold=None):

        metrics = collections.OrderedDict({})

        y_true = df["label"].values.astype(int)

        if f'y_prob{label}' not in cache.keys() :
            y_prob  = model.predict_generator(ds).squeeze() 
            cache[f'y_prob{label}'] = y_prob
        else:
            y_prob = cache[f'y_prob{label}']

        # get roc curve
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        sp_values = np.sqrt(np.sqrt(tpr * (1 - fpr)) * (0.5 * (tpr + (1 - fpr))))
        knee      = np.argmax(sp_values)
        sp_max    = sp_values[knee]

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
                    logger.info("selection inside of the WHO area, by max sp inside of WHO")
                    sp_argmax = np.argmax(sp_values[who_selection])
                    thr       = thresholds[who_selection][sp_argmax]
                else:
                    logger.info("selection by max sp from roc curve")
                    sp_argmax = np.argmax(sp_values)
                    thr       = thresholds[sp_argmax]
 
        else:
            thr = threshold


        y_pred    = (y_prob >= thr).astype(int)
        # confusion matrix and metrics
        conf_matrix                  = confusion_matrix(y_true, y_pred, labels=[False,True])
        tn, fp, fn, tp               = conf_matrix.ravel()
        fa = (fp / (tn + fp)) if (tn+fp) > 0 else 0
        det = (tp / (tp + fn)) if (tp+fn) > 0 else 0 # same as recall or sensibility

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

        # check if the current operation archieve the minimal operation values
        if (metrics["sensitivity"+label]  >= self.min_sensitivity) and (metrics["specificity"+label] >= self.min_specificity):
            metrics["min_spec_sens_reached"]  = True
        else:
            metrics["min_spec_sens_reached"]  = False

        return metrics, thr



class Inference:

    # NOTE: Hard coded configuration fot the inference app
    __metadata = {
        'russia'   : {
            'dataset'   : 'Russia',
            'raw'       : str(DATA_DIR)+'/Russia/russia/raw/images.csv',
            'crop'      : False,
            'blacklist' : str(DATA_DIR)+'/Russia/russia/raw/blacklist.pkl', # NOTE: this is optional configuration
        },
        'china'    : {
            'dataset'   : 'Shenzhen',
            'crop'      : False,
            'raw'       : str(DATA_DIR)+'/Shenzhen/china/raw/Shenzhen_china_table_from_raw.csv',
        },
        'manaus'   : {
            'dataset'   : 'Manaus',
            'crop'      : False,
            'raw'       : str(DATA_DIR)+'/Manaus/manaus/raw/Manaus_manaus_table_from_raw.csv',
        },
        'c_manaus' : {
            'dataset'   : 'Manaus',
            'crop'      : False,
            'raw'       : str(DATA_DIR)+'/Manaus/manaus/raw/Manaus_c_manaus_table_from_raw.csv',
        },
        'imageamento_anonimizado_valid' : {
            'dataset'   : 'SantaCasa',
            'crop'      : False,
            'raw'       : str(DATA_DIR)+'/SantaCasa/imageamento_anonimizado_valid/raw/SantaCasa_imageamento_anonimizado_valid_table_from_raw.csv',
        },
        'caxias'   : {
            'dataset'   : 'Caxias',
            'raw'       : str(DATA_DIR)+'/Caxias/caxias/raw/images.csv',
            'crop'      : False,
            'blacklist' : str(DATA_DIR)+'/Caxias/caxias/raw/blacklist.pkl', # NOTE: this is optional configuration
        },
        'indonesia'   : {
            'dataset'   : 'Indonesia',
            'raw'       : str(DATA_DIR)+'/Indonesia/indonesia/raw/images.csv',
            'crop'      : False,
            'blacklist' : str(DATA_DIR)+'/Indonesia/indonesia/raw/blacklist.pkl', # NOTE: this is optional configuration
        },
        'fiocruz'   : {
            'dataset'   : 'Rio',
            'raw'       : str(DATA_DIR)+'/Rio/fiocruz/raw/Rio_fiocruz_table_from_raw.csv',
            'crop'      : True,
        },
    }


    def __init__(self, key, data_tags, batch_size=32 ):

        self.key = key
        self.batch_size=batch_size
        self.data_tags=data_tags


    def prepare_data(self, tag):
        data_df = pd.read_csv( self.__metadata[tag]['raw'])
        # NOTE: adapt to convnets.data open structure
        data_df.rename(
            columns={
                "image_path": "path",
                "target"    : "label",
            },
            inplace=True,
        )
        dataset = self.__metadata[tag]['dataset']
        path = DATA_DIR / f"{dataset}/{tag}/raw"

        def _append_basepath(row):
            return f"{str(path)}/{row.path}"
        data_df['path'] = data_df.apply(_append_basepath, axis='columns')

        # NOTE: Hack version to remove manually some corrupted images by hand
        if 'blacklist' in self.__metadata[tag]:
            logger.info(f"Applying black list into {tag} dataset...")
            blacklist = pickle.load(open(self.__metadata[tag]['blacklist'], 'rb'))['black_list']
            data_df.drop( index=data_df.loc[data_df['project_id'].isin(blacklist)].index, inplace=True)

        return data_df



    def __call__( self, ctx , output_d):

        train_state = ctx['train_state']
        cache       = ctx['cache']
        threshold   = ctx['threshold']

        model, _, params = convnets.build_model_from_train_state(train_state)

        d  = collections.OrderedDict()
        
        for tag in self.data_tags:
            logger.info(f"Calculating inference for {tag} dataset...")
            data    = self.prepare_data(tag)
            crop    = self.__metadata[tag]['crop']
            ds      = convnets.build_dataset(data, params["image_shape"], batch_size=self.batch_size, crop_header=crop)       
            metrics = self.calculate( ds, data, model, threshold, cache, label=tag )              
            d[tag]  = metrics
            logger.info(tag + f" (Cropped? {'Yes' if crop else 'No'})")
            logger.info( "      SP = %1.2f (Sens = %1.2f, Spec = %1.2f)" % (metrics['sp_index']*100, metrics['sensitivity']*100, metrics['specificity']*100     )    )
            logger.info(f"      tp = {metrics['true_positive']}, tn = {metrics['true_negative']}, fp = {metrics['false_positive']}, fn = {metrics['false_negative']}")

        output_d[self.key] = d



    def calculate( self, ds, df, model, thr, cache, label=''):

        metrics = collections.OrderedDict({})

        y_true  = df["label"].values.astype(int)

        # cache answer
        if f'y_prob_inference_{label}' not in cache.keys() :
            y_prob  = model.predict_generator(ds).squeeze() 
            cache[f'y_prob_inference_{label}'] = y_prob
        else:
            y_prob = cache[f'y_prob_inference_{label}']
    
      
        y_pred    = (y_prob >= thr).astype(int)

        # confusion matrix and metrics
        conf_matrix                  = confusion_matrix(y_true, y_pred, labels=[False,True])
        tn, fp, fn, tp               = conf_matrix.ravel()
        fa = (fp / (tn + fp)) if (tn+fp) > 0 else 0
        det = (tp / (tp + fn)) if (tp+fn) > 0 else 0 # same as recall or sensibility

        # given the threshold
        metrics["sp_index"]       = np.sqrt(np.sqrt(det * (1 - fa)) * (0.5 * (det + (1 - fa))))
        metrics["fa"]             = fa
        metrics["pd"]             = det
        metrics["sensitivity"]    = tp / (tp + fn) if (tp+fn) > 0 else 0 # same as recall
        metrics["specificity"]    = tn / (tn + fp) if (tn+fp) > 0 else 0
        metrics["precision"]      = tp / (tp + fp) if (tp+fp) > 0 else 0
        metrics["recall"]         = tp / (tp + fn) if (tp+fn) > 0 else 0# same as sensibility
        metrics["acc"]            = (tp+tn)/(tp+tn+fp+fn) # accuracy
        metrics["true_negative" ] = tn
        metrics["true_positive" ] = tp
        metrics["false_negative"] = fn
        metrics["false_positive"] = fp

        return metrics



