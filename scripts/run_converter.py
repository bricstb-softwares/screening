#!/usr/bin/env python3

import argparse, os, sys, json, traceback
import tensorflow as tf 

from utils.reprocessing import convert_experiment_to_task

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices)>0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


parser = argparse.ArgumentParser()
parser.add_argument(
    "-e",
    "--experiment_path",
    required=True,
    help="the path of the standalone experiment.",
)

parser.add_argument(
    "--jobs",
    "-j",
    help="Configuration JSON file.",
    default=None,
)

parser.add_argument(
    "--output",
    "-o",
    help="Output file.",
    default=os.getcwd()+'/tuning.pic',
)
args = parser.parse_args()

if len(sys.argv)==1:
    parser.print_help()
    sys.exit(1)
      
try:

    job = json.load(open(args.jobs,'r'))
    test = job['test']
    sort = job['sort']
    convert_experiment_to_task( args.experiment_path, args.output, test, sort)
    sys.exit(0)

except  Exception as e:
    traceback.print_exc()
    sys.exit(1)





