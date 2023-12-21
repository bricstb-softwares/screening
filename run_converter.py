
import argparse, os, sys, luigi


parser = argparse.ArgumentParser()
parser.add_argument(
    "-e",
    "--experiment_path",
    required=True,
    help="the path of the standalone experiment.",
)
parser.add_argument(
    "-o",
    "--output_path",
    help="the output task schema.",
    required=False,
    default='task',
)
args = parser.parse_args()

if len(sys.argv)==1:
    parser.print_help()
    sys.exit(1)
      



from tasks.converter import ConverterPipeline
pipeline = [ConverterPipeline(experiment_path=args.experiment_path, output_path=args.output_path )]
luigi.interface.core.log_level = "WARNING"
luigi.build(pipeline, workers=1, local_scheduler=True)

