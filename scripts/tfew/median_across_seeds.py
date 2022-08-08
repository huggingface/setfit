import argparse
from glob import glob
import json
from os.path import join
from typing import List

from numpy import median
from scipy.stats import iqr

"""
To run: python median_across_seeds.py --path /setfit/scripts/tfew/results/t011b_pretrained/{dataset}
Files are outputted to the directory of the split results.
"""

def get_sample_sizes(path: str) -> List[str]:
    return sorted(list({int(name.split("-")[-2]) for name in glob(f"{path}/train-*-0/seed0")}))


def get_medians_over_seeds(results_path: str) -> None:
    """Given per-split and per-seed T-Few results for some dataset,
    calculates the median score and interquartile range across all seeds,
    and saves them to a `results.json` file in the same path.

    Args:
        path: path to T-Few results: `/setfit/scripts/tfew/results/t011b_pretrained/{dataset}`
    """
    sample_sizes = get_sample_sizes(results_path)

    for sample_size in sample_sizes:
        split_dirs = sorted(glob(join(results_path, f"train-{sample_size}-*")))

        for split_dir in split_dirs:
            seed_results_json =  sorted(glob(join(split_dir, f"seed*/dev_scores.json")))
            seed_metrics = []
            for seed_result_json in seed_results_json:
                with open(seed_result_json) as f:
                    result_dict = json.loads(f.readlines()[-1])
                seed_metrics.append(result_dict["accuracy"])

            with open(join(split_dir, 'results.json'), 'w') as f:
                json.dump({"score": median(seed_metrics), "iqr": iqr(seed_metrics)}, f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    args = parser.parse_args()

    get_medians_over_seeds(args.path)


if __name__ == "__main__":
    main()
