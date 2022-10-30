import argparse
import json
from os import listdir
from os.path import isdir, isfile, join

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_folder",
        "-e",
        required=True,
        type=str,
        help="The folder path of the experiment created by `run_fewshot.py`.",
    )

    args = parser.parse_args()

    return args


def get_folders(folder):
    return [join(folder, f) for f in listdir(folder) if isdir(join(folder, f))]


if __name__ == "__main__":

    args = parse_args()

    dataset_folders = get_folders(args.exp_folder)
    for dataset_folder in dataset_folders:
        run_folders = get_folders(dataset_folder)

        scores = []
        for run_folder in run_folders:
            with open(join(run_folder, "results.json"), "r") as f:
                score = json.load(f)["score"]
                scores.append(score)

        scores = np.array(scores)
        with open(join(dataset_folder, "results.json"), "w") as f:
            json.dump(
                {
                    "mean": np.mean(scores).item(),
                    "std": np.std(scores).item(),
                },
                f,
            )
