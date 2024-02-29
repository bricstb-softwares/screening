

import collections
from screening.validation.crossval import crossval_table
from pprint import pprint

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


#extra_suf='_val'
extra_suf=''
conf_dict = collections.OrderedDict(
    {
        #'loose'   : create_op_dict( 'loose'  , extra_suf=extra_suf ),
        'medium'  : create_op_dict( 'medium' , extra_suf=extra_suf),
        #'tight'   : create_op_dict( 'tight'  , extra_suf=extra_suf),
    }
)
pprint(conf_dict)
models = [
    ( 'user.philipp.gaspar.convnets_v1.baseline.shenzhen_santacasa.exp.20240207.r1'                          , 'base.sh-sc.e'        ),
    ( 'user.philipp.gaspar.convnets_v1.altogether.shenzhen_santacasa.exp_wgan_p2p.20240207.r1'               , 'alto.sh-sc.ewp'      ),
    ( 'user.philipp.gaspar.convnets_v1.interleaved.shenzhen_santacasa.exp_wgan_p2p.20240207.r1'              , 'inte.sh-sc.ewp'      ),
    #( 'user.philipp.gaspar.convnets_v1.altogether.shenzhen_santacasa.exp_wgan_p2p_cycle.a19a3a4f8c.r1'         , 'alto.sh-sc.ewpc'     ),
    #( 'user.philipp.gaspar.convnets_v1.interleaved.shenzhen_santacasa.exp_wgan_p2p_cycle.a19a3a4f8c.r1'        , 'inte.sh-sc.ewpc'     ),
]
models_plus_manaus = [
    # plus manaus
    #( 'user.philipp.gaspar.convnets_v1.baseline.shenzhen_santacasa_manaus.exp.ffe6cbee11.r1'                   , 'base.sh-sc-ma.e'     ),
    #( 'user.philipp.gaspar.convnets_v1.altogether.shenzhen_santacasa_manaus.exp_wgan_p2p.0d13030165.r1'        , 'alto.sh-sc-ma.ewp'   ),
    #( 'user.philipp.gaspar.convnets_v1.interleaved.shenzhen_santacasa_manaus.exp_wgan_p2p.ac79954ba0.r1'       , 'inte.sh-sc-ma.ewp'   ),
    #( 'user.philipp.gaspar.convnets_v1.altogether.shenzhen_santacasa_manaus.exp_wgan_p2p_cycle.c5143abd1b.r1'  , 'alto.sh-sc-ma.ewpc'  ),
    #( 'user.philipp.gaspar.convnets_v1.interleaved.shenzhen_santacasa_manaus.exp_wgan_p2p_cycle.c5143abd1b.r1' , 'inte.sh-sc-ma.ewpc'  ),
]
basepath='/mnt/brics_data/models/v1'
for path, train_tag in models+models_plus_manaus:
    cv = crossval_table( conf_dict )
    cv.fill( basepath+'/'+path , train_tag )
    table = cv.table
    best_sorts = cv.filter_sorts( table,      'sp_index_op'    )
    best_tests = cv.filter_tests( best_sorts, 'sp_index_test' )
    cv.report(table, best_sorts, best_tests, train_tag, train_tag )

#cv = crossval_table( conf_dict )
#for path, train_tag in models:
#    cv.fill( basepath+'/'+path , train_tag )
#table      = cv.table
#best_sorts = cv.filter_sorts( table,      'sp_index_op'    )
#best_tests = cv.filter_tests( best_sorts, 'sp_index_test' )
#cv.report(table, best_sorts, best_tests, 'shenzhen_santacasa', 'shenzhen_santacasa' , detailed=False)
#
#cv = crossval_table( conf_dict )
#for path, train_tag in models_plus_manaus:
#    cv.fill( basepath+'/'+path , train_tag )
#table      = cv.table
#best_sorts = cv.filter_sorts( table,      'sp_index_op'    )
#best_tests = cv.filter_tests( best_sorts, 'sp_index_test' )
#cv.report(table, best_sorts, best_tests, 'shenzhen_santacasa_manaus', 'shenzhen_santacasa_manaus' , detailed=False )



