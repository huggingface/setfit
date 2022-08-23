import argparse
import json
import os
import tarfile
from glob import glob
from os.path import splitext
from typing import List, Tuple

from numpy import mean, std


"""
To run: python create_summary_table.py --path ~/setfit/scripts/{method_name}/{results}/{model_name}
e.g. python create_summary_table.py --path ~/setfit/scripts/tfew/results/t03b_pretrained
or: python create_summary_table.py --path scripts/{method_name}/{model_name}.tar.gz
Files are outputted to the directory of the results.
"""


def extract_results(path: str) -> None:
    tar = tarfile.open(path, "r:gz")
    unzip_path = splitext(splitext(path)[-2])[-2]
    tar.extractall(path=os.path.dirname(unzip_path))
    tar.close()
    return unzip_path


def get_sample_sizes(path: str) -> List[str]:
    return sorted(list({int(name.split("-")[-2]) for name in glob(f"{path}/*/train-*-0")}))


def get_formatted_ds_metrics(path: str, dataset: str, sample_sizes: List[str]) -> Tuple[str, List[str]]:
    formatted_row = []
    
    for sample_size in sample_sizes:
        result_jsons = sorted(glob(os.path.join(path, dataset, f"train-{sample_size}-*", "results.json")))
        split_metrics = []
        assert len(split_metrics) > 0 
        for result_json in result_jsons:
            with open(result_json) as f:
                result_dict = json.load(f)

            
            metric_name = result_dict.get("measure", "N/A")
            split_metrics.append(result_dict["score"] * 100)
        formatted_row.extend([f"{mean(split_metrics):.2f}", f"{std(split_metrics):.2f}"])

    return metric_name, formatted_row


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

    sample_sizes = get_sample_sizes(unzipped_path)
    header_row = ["dataset", "measure"]
    for sample_size in sample_sizes:
        header_row.append(f"{sample_size}_avg")
        header_row.append(f"{sample_size}_std")

    csv_lines = [header_row]
    for dataset in os.listdir(unzipped_path):
        metric_name, formatted_metrics = get_formatted_ds_metrics(unzipped_path, dataset, sample_sizes)
        dataset_row = [dataset, metric_name, *formatted_metrics]
        csv_lines.append(dataset_row)

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
