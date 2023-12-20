
import argparse, os, sys, luigi


parser = argparse.ArgumentParser()
parser.add_argument(
    "--experiment_path",
    required=True,
    help="the path of the standalone experiment.",
)
parser.add_argument(
    "--task_output",
    help="the output task schema.",
    required=False,
    default='task',
)
args = parser.parse_args()

if len(sys.argv)==1:
    parser.print_help()
    sys.exit(1)
      



from pipelines.converter import BaselinePipeline
pipeline = [BaselinePipeline(experiment_path=args.experiment_path, output_path=args.task_output )]
luigi.interface.core.log_level = "WARNING"
luigi.build(pipeline, workers=1, local_scheduler=True)

#task = BaselineCnv(
#            experiment_path = args.experiment_path, 
#            output_path     = args.task_output
#        )
#task.requires()
#task.run()
