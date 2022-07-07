import json
from typing import Tuple

from datasets import Dataset


DEV_DATASET_TO_METRIC = {
    "sst2": "accuracy",
    "imdb": "accuracy",
    "subj": "accuracy",
    "bbc-news": "accuracy",
    "enron_spam": "accuracy",
    "student-question-categories": "accuracy",
    "TREC-QC": "accuracy",
    "toxic_conversations": "matthews_correlation",
}

TEST_DATASET_TO_METRIC = {
    "emotion": "accuracy",
    "SentEval-CR": "accuracy",
    "sst5": "accuracy",
    "ag_news": "accuracy",
    "amazon_counterfactual_en": "matthews_correlation",
}


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
