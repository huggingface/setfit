import argparse
import json
import os
import string
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


"""
To run:
python plot_summary_comparison.py --paths scripts/{method_name}/results/{model_name}
Multiple paths can be provided. The produced plots are outputted to scripts/images/v_{id}/{dataset}.png.

See https://github.com/huggingface/setfit/pull/268#issuecomment-1434549208 for an example of the plots
produced by this script.
"""


def get_sample_sizes(path: str) -> List[str]:
    return sorted(list({int(name.split("-")[-2]) for name in glob(f"{path}/*/train-*-0")}))


def get_formatted_ds_metrics(path: str, dataset: str, sample_sizes: List[str]) -> Tuple[str, List[str]]:
    split_metrics = defaultdict(list)

    for sample_size in sample_sizes:
        result_jsons = sorted(glob(os.path.join(path, dataset, f"train-{sample_size}-*", "results.json")))
        for result_json in result_jsons:
            with open(result_json) as f:
                result_dict = json.load(f)

            metric_name = result_dict.get("measure", "N/A")
            split_metrics[sample_size].append(result_dict["score"])

    return metric_name, split_metrics


def plot_summary_comparison(paths: List[str]) -> None:
    """Given a list of paths to output directories produced by e.g. `scripts/setfit/run_fewshot.py`,
    produce and save boxplots that compare the various results.

    The plots are saved to scripts/images/v_{id}/{dataset}.png, i.e. one plot per dataset.

    Args:
        paths (List[str]): List of paths to output directories, generally
            `scripts/{method_name}/results/{model_name}`
    """

    # Parse the result paths
    dataset_to_df = defaultdict(pd.DataFrame)
    dataset_to_metric = {}
    for path_index, path in enumerate(paths):
        ds_to_metric, this_dataset_to_df = get_summary_df(path)
        for dataset, df in this_dataset_to_df.items():
            df["path_index"] = path_index
            dataset_to_df[dataset] = pd.concat((dataset_to_df[dataset], df))
        dataset_to_metric = dataset_to_metric | ds_to_metric

    # Prepare folder for storing figures
    image_dir = Path("scripts") / "images"
    image_dir.mkdir(exist_ok=True)
    new_version = (
        max([int(path.name[2:]) for path in image_dir.glob("v_*/") if path.name[2:].isdigit()], default=0) + 1
    )
    output_dir = image_dir / f"v_{new_version}"
    output_dir.mkdir()

    # Create the plots per each dataset
    for dataset, df in dataset_to_df.items():
        columns = [column for column in df.columns if not column.startswith("path")]
        fig, axes = plt.subplots(ncols=len(columns), sharey=True)
        for column_index, column in enumerate(columns):
            ax = axes[column_index]

            # Set the y label only for the first column
            if column_index == 0:
                ax.set_ylabel(dataset_to_metric[dataset])

            # Set positions to 0, 0.25, ..., one position per boxplot
            # This places the boxplots closer together
            n_boxplots = len(df["path_index"].unique())
            allotted_box_width = 0.2
            positions = [allotted_box_width * i for i in range(n_boxplots)]
            ax.set_xlim(-allotted_box_width * 0.75, allotted_box_width * (n_boxplots - 0.25))

            df[[column, "path_index"]].groupby("path_index", sort=True).boxplot(
                subplots=False, ax=ax, column=column, positions=positions
            )

            k_shot = column.split("-")[-1]
            ax.set_xlabel(f"{k_shot}-shot")
            if n_boxplots > 1:
                # If there are multiple boxplots, override the labels at the bottom generated by pandas
                if n_boxplots <= 26:
                    ax.set_xticklabels(string.ascii_uppercase[:n_boxplots])
                else:
                    ax.set_xticklabels(range(n_boxplots))
            else:
                # Otherwise, just remove the xticks
                ax.tick_params(labelbottom=False)

        if n_boxplots > 1:
            fig.suptitle(
                f"Comparison between various baselines on the {dataset}\ndataset under various $K$-shot conditions"
            )
        else:
            fig.suptitle(f"Results on the {dataset} dataset under various $K$-shot conditions")
        fig.tight_layout()
        plt.savefig(str(output_dir / dataset))


def get_summary_df(path: str) -> None:
    """Given per-split results, return a mapping from dataset to metrics (e.g. "accuracy") and
    a mapping from dataset to pandas DataFrame that stores the results

    Args:
        path: path to per-split results: generally `scripts/{method_name}/results/{model_name}`,
    """

    sample_sizes = get_sample_sizes(path)
    header_row = ["dataset", "measure"]
    for sample_size in sample_sizes:
        header_row.append(f"{sample_size}_avg")
        header_row.append(f"{sample_size}_std")

    dataset_to_metric = {}
    dataset_to_df = {}
    for dataset in next(os.walk(path))[1]:
        metric_name, split_metrics = get_formatted_ds_metrics(path, dataset, sample_sizes)
        dataset_df = pd.DataFrame(split_metrics.values(), index=[f"{dataset}-{key}" for key in split_metrics]).T
        dataset_to_metric[dataset] = metric_name
        dataset_to_df[dataset] = dataset_df
    return dataset_to_metric, dataset_to_df


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--paths", nargs="+", type=str)
    args = parser.parse_args()

    if args.paths:
        plot_summary_comparison(args.paths)
    else:
        raise Exception("Please provide at least one path via the `--paths` CLI argument.")


if __name__ == "__main__":
    main()
