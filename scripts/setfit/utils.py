from typing import List, Tuple

from datasets import Dataset, DatasetDict, load_dataset

from setfit.data import create_fewshot_splits, create_fewshot_splits_multilabel


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

MULTILINGUAL_DATASET_TO_METRIC = {
    f"amazon_reviews_multi_{lang}": "mae" for lang in ["en", "de", "es", "fr", "ja", "zh"]
}


def load_data_splits(dataset: str, sample_sizes: List[int]) -> Tuple[DatasetDict, Dataset]:
    """Loads a dataset from the Hugging Face Hub and returns the test split and few-shot training splits."""
    print(f"\n\n\n============== {dataset} ============")
    # Load one of the SetFit training sets from the Hugging Face Hub
    train_split = load_dataset(f"SetFit/{dataset}", split="train")
    train_splits = create_fewshot_splits(train_split, sample_sizes)
    test_split = load_dataset(f"SetFit/{dataset}", split="test")
    print(f"Test set: {len(test_split)}")
    return train_splits, test_split


def load_data_splits_multilabel(dataset: str, sample_sizes: List[int]) -> Tuple[DatasetDict, Dataset]:
    """Loads a dataset from the Hugging Face Hub and returns the test split and few-shot training splits."""
    print(f"\n\n\n============== {dataset} ============")
    # Load one of the SetFit training sets from the Hugging Face Hub
    train_split = load_dataset(f"SetFit/{dataset}", split="train")
    train_splits = create_fewshot_splits_multilabel(train_split, sample_sizes)
    test_split = load_dataset(f"SetFit/{dataset}", split="test")
    print(f"Test set: {len(test_split)}")
    return train_splits, test_split
