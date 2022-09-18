import argparse
import json
import os
import tarfile
from collections import defaultdict
from glob import glob
from os import listdir
from os.path import isdir, join, splitext
from typing import List, Tuple

from numpy import mean, median, std
from scipy.stats import iqr


"""
To run: python create_summary_table.py --path scripts/{method_name}/{results}/{model_name}
or: python create_summary_table.py --path scripts/{method_name}/{model_name}.tar.gz
Files are outputted to the directory of the results.
"""

TEST_DATASET_TO_METRIC = {
    "emotion": "accuracy",
    "SentEval-CR": "accuracy",
    "sst5": "accuracy",
    "ag_news": "accuracy",
    "enron_spam": "accuracy",
    "amazon_counterfactual_en": "matthews_correlation",
}


def extract_results(path: str) -> None:
    tar = tarfile.open(path, "r:gz")
    unzip_path = splitext(splitext(path)[-2])[-2]
    tar.extractall(path=os.path.dirname(unzip_path))
    tar.close()
    return unzip_path


def get_sample_sizes(path: str) -> List[str]:
    return sorted(list({int(name.split("-")[-2]) for name in glob(f"{path}/*/train-*-0")}))


def get_tfew_sample_sizes(path: str) -> List[str]:
    return sorted(list({int(name.split("-")[-2]) for name in glob(f"{path}/train-*-0/seed0")}))


def compute_tfew_medians(results_path: str) -> None:
    """Given per-split and per-seed T-Few results for multiple dataset,
    calculates the median score and interquartile range across all seeds,
    and saves them to a `results.json` file in the same path.

    Args:
        results_path: path to T-Few results: `/setfit/scripts/tfew/results/t03b_pretrained`
    """

    for dataset in listdir(results_path):
        dataset_path = join(results_path, dataset)
        if isdir(dataset_path):
            dataset_metric = TEST_DATASET_TO_METRIC[dataset]
            sample_sizes = get_tfew_sample_sizes(dataset_path)

            for sample_size in sample_sizes:
                split_dirs = sorted(glob(join(dataset_path, f"train-{sample_size}-*")))
                assert split_dirs is not None

                for split_dir in split_dirs:
                    seed_results_json = sorted(glob(join(split_dir, "seed*/dev_scores.json")))
                    seed_metrics = []
                    for seed_result_json in seed_results_json:
                        with open(seed_result_json) as f:
                            result_dict = json.loads(f.readlines()[-1])
                        seed_metrics.append(result_dict[dataset_metric] * 100)

                    with open(join(split_dir, "results.json"), "w") as f:
                        json.dump(
                            {"score": median(seed_metrics), "measure": dataset_metric, "iqr": iqr(seed_metrics)}, f
                        )


def get_formatted_ds_metrics(path: str, dataset: str, sample_sizes: List[str]) -> Tuple[str, List[str]]:
    formatted_row = []
    exact_metrics, exact_stds = {}, {}

    for sample_size in sample_sizes:
        result_jsons = sorted(glob(os.path.join(path, dataset, f"train-{sample_size}-*", "results.json")))
        split_metrics = []

        for result_json in result_jsons:
            with open(result_json) as f:
                result_dict = json.load(f)

            metric_name = result_dict.get("measure", "N/A")
            split_metrics.append(result_dict["score"])

        exact_metrics[sample_size] = mean(split_metrics)
        exact_stds[sample_size] = std(split_metrics)
        formatted_row.extend([f"{exact_metrics[sample_size]:.1f}", f"{exact_stds[sample_size]:.1f}"])

    return metric_name, formatted_row, exact_metrics, exact_stds, sample_sizes


def create_summary_table(results_path: str) -> None:
    """Given per-split results, creates a summary table of all datasets,
    with average metrics and standard deviations.

    Args:
        path: path to per-split results: either `scripts/{method_name}/{results}/{model_name}`,
            or `final_results/{method_name}/{model_name}.tar.gz`
    """

    if results_path.endswith("tar.gz"):
        unzipped_path = extract_results(results_path)
    else:
        unzipped_path = results_path

    if "tfew" in unzipped_path:
        print("Computing medians for T-Few...")
        compute_tfew_medians(unzipped_path)

    sample_sizes = get_sample_sizes(unzipped_path)
    header_row = ["dataset", "measure"]
    for sample_size in sample_sizes:
        header_row.append(f"{sample_size}_avg")
        header_row.append(f"{sample_size}_std")

    csv_lines = [header_row]

    means, stds = defaultdict(list), defaultdict(list)
    for dataset in next(os.walk(unzipped_path))[1]:
        metric_name, formatted_metrics, exact_metrics, exact_stds, sample_sizes = get_formatted_ds_metrics(
            unzipped_path, dataset, sample_sizes
        )
        dataset_row = [dataset, metric_name, *formatted_metrics]
        csv_lines.append(dataset_row)

        # Collect exact metrics for overall average and std calculation
        for sample_size in sample_sizes:
            means[sample_size].append(exact_metrics[sample_size])
            stds[sample_size].append(exact_stds[sample_size])

    # Generate row for overall average
    formatted_average_row = []
    for sample_size in sample_sizes:
        overall_average = mean(means[sample_size])
        overall_std = mean(stds[sample_size])
        formatted_average_row.extend([f"{overall_average:.1f}", f"{overall_std:.1f}"])
    csv_lines.append(["Average", "N/A", *formatted_average_row])

    output_path = os.path.join(unzipped_path, "summary_table.csv")
    print("=" * 80)
    print("Summary table:\n")
    with open(output_path, "w") as f:
        for line in csv_lines:
            f.write(",".join(line) + "\n")
            print(", ".join(line))
    print("=" * 80)
    print(f"Saved summary table to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type=str)
    args = parser.parse_args()

    create_summary_table(args.path)


if __name__ == "__main__":
    main()
