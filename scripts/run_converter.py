#!/usr/bin/env python


import os, sys, pickle, json, traceback
import argparse
import pandas as pd
import tensorflow as tf

from loguru import logger
from screening.utils import commons
from screening.utils.data import prepare_data
from screening.utils.convnets import (
    prepare_model,
    build_model_from_train_state,
    save_job_state,
)
from screening.validation.evaluate import (
    evaluate,
)

from screening.utils.data import split_dataframe


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
    job_path = output_path + '/output.pkl'
 
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


def run():

    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices)>0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.run_functions_eagerly(False)

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment_path", required=True, help="the path of the standalone experiment.")
    parser.add_argument("--jobs","-j",help="Configuration JSON file.",default=None)
    parser.add_argument("--output","-o",help="Output file.",default=os.getcwd())
    args = parser.parse_args()

    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)

    dry_run = os.environ.get('JOB_DRY_RUN', 'false') == 'true'

    if dry_run:
        sys.exit(0)

    if os.path.exists( args.output+'/output.pkl' ):
        sys.exit(0)

    try:
        job = json.load(open(args.jobs,'r'))
        test = job['test']
        sort = job['sort']
        convert_experiment_to_task( args.experiment_path, args.output, test, sort)
        sys.exit(0)
    except  Exception as e:
        traceback.print_exc()
        sys.exit(1)





if __name__ == "__main__":
    run()
