import argparse
import json

import luigi
from pipelines.report import ReportPipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_hash",
        "-hash",
        required=True,
        help="name and hash of experiment.",
    )
    parser.add_argument(
        "--report_info",
        "-info",
        help="Report JSON file.",
        default="report_info.json",
    )
    args = parser.parse_args()

    with open(args.report_info, "rt") as file:
        d_args = argparse.Namespace()
        d_args.__dict__.update(json.load(file))

    dataset_info = vars(d_args)

    pipeline = [ReportPipeline(args.experiment_hash, dataset_info)]
    luigi.build(pipeline, workers=1, local_scheduler=True)
