import json
from typing import Tuple

from datasets import Dataset


def get_label_mappings(dataset: Dataset) -> Tuple[int, dict, dict]:
    """Returns the label mappings of the dataset."""
    label_ids = dataset.unique("label")
    label_names = dataset.unique("label_text")
    label2id = {label: idx for label, idx in zip(label_names, label_ids)}
    id2label = {idx: label for label, idx in label2id.items()}
    num_labels = len(label_ids)
    return num_labels, label2id, id2label


def save_metrics(metrics: dict, metrics_filepath):
    with open(metrics_filepath, "w") as f:
        json.dump(metrics, f)
