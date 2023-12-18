import argparse
import json
import os, sys, traceback

import luigi
from pipelines.train import (
    AltogetherPipeline,
    BaselineFineTuningPipeline,
    BaselinePipeline,
    InterleavedPipeline,
    SyntheticPipeline,
)

import tensorflow as tf 
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices)>0:
    tf.config.experimental.set_memory_growth(physical_devices[0], False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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

    args = parser.parse_args()


    try: 

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
                job['tracking_url'] = os.environ.get("TRACKING_URL", "")
                job['job_id'] = int(os.environ.get("JOB_ID", "-1"))
                job['run_id'] = os.environ.get('TRACKING_RUN_ID', "")
                job['dry_run'] = os.environ.get('JOB_DRY_RUN', "false") == 'true'
                job['device']  = int(os.environ.get('CUDA_VISIBLE_DEVICES','-1'))
                task_params["job"] = job
        else:
            task_params["job"] = {}

        processes = {
            "baseline": BaselinePipeline,
            "synthetic": SyntheticPipeline,
            "interleaved": InterleavedPipeline,
            "altogether": AltogetherPipeline,
            "baseline_fine_tuning": BaselineFineTuningPipeline,
        }

        pipeline = [processes[args.process_name](**task_params)]
        luigi.build(pipeline, workers=1, local_scheduler=True)

        sys.exit(0)

    except  Exception as e:
        traceback.print_exc()
        sys.exit(1)