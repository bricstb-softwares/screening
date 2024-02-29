#!/usr/bin/env python


import argparse, json, os, sys, traceback, luigi
import tensorflow as tf 
from screening.pipelines import get_task
from loguru import logger

def run():
    
 
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name", "-m", help="Name of the model.", required=True
    )
    parser.add_argument(
        "--process_name", "-p", help="Name of the process.", required=True
    )
    parser.add_argument(
        "--dataset_info",
        "-info",
        help="Configuration JSON file.",
        default="dataset_info.json",
    )
    parser.add_argument(
        "--hyperparameters",
        "-params",
        help="Configuration JSON file.",
        default="hyperparameters.json",
    )

    parser.add_argument(
        "--jobs",
        "-j",
        help="Configuration JSON file.",
        default=None,
    )

    parser.add_argument(
        "--parent_task",
        help="The parent task used for finetuning method. Only use this argument if you are in job mode.",
        default=None,
    )

    args = parser.parse_args()


    try: 

        dry_run      = os.environ.get("JOB_DRY_RUN", "false")=="true"
        job_basepath = os.environ.get("JOB_WORKAREA", os.getcwd())
        job_output   = job_basepath+'/output.pkl'

        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices)>0:
            logger.info("GPU available!")
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            tf.config.run_functions_eagerly(False)
        else:
            logger.warning("only CPU available.")

        with open(args.dataset_info, "rt") as file:
            d_args = argparse.Namespace()
            d_args.__dict__.update(json.load(file))

        with open(args.hyperparameters, "rt") as file:
            h_args = argparse.Namespace()
            h_args.__dict__.update(json.load(file))

        task_params = vars(h_args)
        task_params["dataset_info"] = vars(d_args)
        if args.jobs:
            with open(args.jobs, "r") as file:
                job = json.load(file)
                job['parent'] = args.parent_task
                task_params["job"] = job
        else:
            task_params["job"] = {}

   

        model_name = task_params.pop("model_name")
        task       = get_task(model_name, args.process_name )

        if dry_run:
            if "epochs" in task_params.keys():
                logger.info("setting epochs to 5 since this is a dry_run job.")
                task_params["epochs"]=5

            if os.path.exists(job_output):
                logger.info(f"removing {job_output} since this is a dry_run job.")
                os.system(f"rm -rf {job_output}")
            

        pipeline = [task(**task_params)]
        luigi.build(pipeline, workers=1, local_scheduler=True)
        
        if dry_run:
            if os.path.exists(job_output):
                logger.info(f"removing {job_output} since this is a dry_run job.")
                os.system(f"rm -rf {job_output}")
            
        sys.exit(0)

    except  Exception as e:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run()

