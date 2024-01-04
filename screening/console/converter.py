


import os, sys, pickle
import pandas as pd
from loguru import logger
from utils import commons

from utils.data import prepare_data

from utils.convnets import (
    split_dataframe,
    prepare_model,
    build_model_from_train_state,
    save_job_state,
)

from utils.validation import (
    evaluate,
)


def convert_experiment_to_task( experiment_path : str, output_path : str, test : int, sort : int , seed=42):

    # reading all parameters from the current task
    task_params    = pickle.load(open(experiment_path+'/task_params.pkl','rb'))
    dataset_info   = task_params['dataset_info']


    data_list   = []
    # prepare data
    for dataset in dataset_info:
        for source in dataset_info[dataset]["sources"]:
            tag   = dataset_info[dataset]["tag"]
            files = dataset_info[dataset]["sources"][source]
            logger.info(f"Readinf data from {tag}...")
            d     = prepare_data( source, dataset, tag, files )
            data_list.append(d)

    data = pd.concat(data_list)
    data = data.sample(frac=1, random_state=seed)   

    experiment_type = task_params['experiment_id'].split('_')[0].lower().replace('train','')
    experiment_hash = task_params['experiment_id'].split('_')[1]

    # get training control tags
    logger.info(f"Converting experiment with hash {experiment_hash}")

    logger.info(f"Converting Fold #{test} / Validation #{sort}...")

    #folder_path = f'{output_path}/job.test_{test}.sort_{sort}'
    #commons.create_folder(folder_path)
    
    job_path = output_path + '/output.pkl'
    #if os.path.exists(job_path):
    #    logger.warning(f'job output exist into {output_path}')
    #    return False
        
    logger.info(f"Experiment type is {experiment_type}")


    if experiment_type in ['baseline']:
        train_data  = split_dataframe(data, test, sort, "train_real")
    elif experiment_type in ['interleaved', 'altogether']:
        train_real = split_dataframe(data, test, sort, "train_real")
        train_fake = split_dataframe(data, test, sort, "train_fake")
        train_data = pd.concat([train_real, train_fake])
    else:
        RuntimeError(f"Experiment ({experiment_type}) type not implemented")
    
    
    valid_data  = split_dataframe(data, test, sort, "valid_real")
    test_data   = split_dataframe(data, test, sort, "test_real" )

    # build the model
    model_path   = experiment_path +f"/cnn_fold{test}/sort{sort}/"
    logger.info(f"Reading model from {model_path}...")
    model, history, params = build_model_from_train_state( model_path )
    train_state =  prepare_model(model, history, params)

    # get train state
    logger.info("Applying the validation...")
    train_state = evaluate(train_state, train_data, valid_data, test_data)

    logger.info(f"Saving job state for test {test} and sort {sort} into {job_path}")
    save_job_state( job_path,
                    train_state, 
                    test = test,
                    sort = sort,
                    metadata = {
                        'task_params' : task_params,
                        'hash'        : experiment_hash,
                        'type'        : experiment_type,
                    })

   
    logger.info("")
    logger.info(f"=== End: '{__name__}' ===")
    logger.info("")

    return True

