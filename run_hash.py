import argparse
import json
import os, sys, traceback

import luigi
from pipelines.train import processes
from pprint import pprint
import collections

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


    args = parser.parse_args()


    try: 

        with open(args.dataset_info, "rt") as file:
            d_args = argparse.Namespace()
            d = json.load(file)
            d_args.__dict__.update(collections.OrderedDict(d))

        with open(args.hyperparameters, "rt") as file:
            h_args = argparse.Namespace()
            h_args.__dict__.update(json.load(file))

        task_params = vars(h_args)
        task_params["dataset_info"] = vars(d_args)
        task_params["job"] = {}


        task = next(processes[args.process_name](**task_params).requires())
        print(task.get_hash())
        sys.exit(0)

    except  Exception as e:
        traceback.print_exc()
        sys.exit(1)