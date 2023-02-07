from contextlib import contextmanager
from dataclasses import dataclass, field
import json
from time import monotonic_ns
from typing import Any, Dict, List, NamedTuple, Tuple
from collections import Counter

from datasets import Dataset, DatasetDict, load_dataset
from sentence_transformers import losses, InputExample
import numpy as np

from .data import create_fewshot_splits, create_fewshot_splits_multilabel
from .modeling import SupConLoss, sentence_pairs_generation

from random import shuffle

SEC_TO_NS_SCALE = 1000000000


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
    "enron_spam": "accuracy",
    "amazon_counterfactual_en": "matthews_correlation",
}

MULTILINGUAL_DATASET_TO_METRIC = {
    f"amazon_reviews_multi_{lang}": "mae" for lang in ["en", "de", "es", "fr", "ja", "zh"]
}

LOSS_NAME_TO_CLASS = {
    "CosineSimilarityLoss": losses.CosineSimilarityLoss,
    "ContrastiveLoss": losses.ContrastiveLoss,
    "OnlineContrastiveLoss": losses.OnlineContrastiveLoss,
    "BatchSemiHardTripletLoss": losses.BatchSemiHardTripletLoss,
    "BatchAllTripletLoss": losses.BatchAllTripletLoss,
    "BatchHardTripletLoss": losses.BatchHardTripletLoss,
    "BatchHardSoftMarginTripletLoss": losses.BatchHardSoftMarginTripletLoss,
    "SupConLoss": SupConLoss,
}


def default_hp_space_optuna(trial) -> Dict[str, Any]:
    from transformers.integrations import is_optuna_available

    assert is_optuna_available(), "This function needs Optuna installed: `pip install optuna`"
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "num_epochs": trial.suggest_int("num_epochs", 1, 5),
        "num_iterations": trial.suggest_categorical("num_iterations", [5, 10, 20]),
        "seed": trial.suggest_int("seed", 1, 40),
        "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16, 32, 64]),
    }

def create_pairs_dataset(dataset, num_iterations: int=20) -> Dict[str, Dataset]:
    # Randomly select `num_pairs` pairs from the test split,
    # and assign them Yes (in-class) / No (out-of-class) labels.

    x_train = dataset["text"]
    y_train = dataset["label"]
    examples = []

    for _ in range(num_iterations):
        pair_examples = sentence_pairs_generation(
            np.array(x_train), np.array(y_train), examples
        )

    # Construct Dataset from examples
    text_a_list, text_b_list, labels = [], [], []
    for example in pair_examples:
        text_a_list.append(example.texts[0])
        text_b_list.append(example.texts[1])
        labels.append(example.label)

    pairs_dataset = Dataset.from_dict(dict(text_a=text_a_list, text_b=text_b_list, label=labels))
    return pairs_dataset

def load_pseudolabeled_examples(pseudolabels_json: str, top_n: int) -> List[InputExample]:
    pseudolabels, labels, entropies, examples = [], [], [], []

    with open(pseudolabels_json) as f:
        rows = list(f)
        selected_rows = rows[:top_n] #rows[:top_n // 2] + rows[-top_n // 2:]
        # shuffle(selected_rows)
        for row in selected_rows:
            data = json.loads(row)
            pseudolabels.append(data["pred"])
            labels.append(data["label"])
            examples.append(InputExample(texts=[data["text_a"], data["text_b"]], label=float(data["pred"])))

    matching = [a == b for a, b in zip(pseudolabels, labels)]
    pl_accuracy = sum(matching) / len(matching)
    
    print(f"Loaded {len(examples)} pseudolabels with accuracy: {pl_accuracy:.4f}")
    return examples, pl_accuracy


def label_proportions(data):
    counts = Counter(data['label']).items()
    total = sum(i[1] for i in counts)
    sorted_counts = sorted(counts, key=lambda item: item[1])
    return [f"{item[0]}: {(item[1] / total) * 100:.2f}" for item in sorted_counts]


def test_splits():
   [load_data_splits(ds, sample_sizes=[0], n=300) for ds in TEST_DATASET_TO_METRIC.keys()]


def load_data_splits(
    dataset: str, sample_sizes: List[int], add_data_augmentation: bool = False, n: int = None
    ) -> Tuple[DatasetDict, Dataset]:
    """Loads a dataset from the Hugging Face Hub and returns the test split and few-shot training splits."""
    print(f"\n\n\n============== {dataset} ============")
    # Load one of the SetFit training sets from the Hugging Face Hub
    train_split = load_dataset(f"SetFit/{dataset}", split="train")
    train_splits, unlabeled_splits = create_fewshot_splits(train_split, sample_sizes, add_data_augmentation, dataset)
    test_split = load_dataset(f"SetFit/{dataset}", split="test")

    print(f"\n**** {dataset}  - Class counts ****")
    print(f"Original train split: {label_proportions(train_split)}")
    print(f"Original test split: {label_proportions(test_split)}")

    # Debug class balance
    # for split_name, split in list(unlabeled_splits.items())[::2]:
    #     print(f"{split_name}: {label_proportions(split)}")
    # print()
    # print("sample sizes:", sample_sizes)
    # if n is not None:
    #     for split_name, split in list(unlabeled_splits.items())[::2]:
    #         print(f"{split_name}: {label_proportions(split[:n])}")
    #     print()

    print(f"Test set: {len(test_split)}")
    return train_splits, test_split, unlabeled_splits


def load_data_splits_multilabel(dataset: str, sample_sizes: List[int]) -> Tuple[DatasetDict, Dataset]:
    """Loads a dataset from the Hugging Face Hub and returns the test split and few-shot training splits."""
    print(f"\n\n\n============== {dataset} ============")
    # Load one of the SetFit training sets from the Hugging Face Hub
    train_split = load_dataset(f"SetFit/{dataset}", "multilabel", split="train")
    train_splits = create_fewshot_splits_multilabel(train_split, sample_sizes)
    test_split = load_dataset(f"SetFit/{dataset}", "multilabel", split="test")
    print(f"Test set: {len(test_split)}")
    return train_splits, test_split


@dataclass
class Benchmark:
    """
    Performs simple benchmarks of code portions (measures elapsed time).

        Typical usage example:

        bench = Benchmark()
        with bench.track("Foo function"):
            foo()
        with bench.track("Bar function"):
            bar()
        bench.summary()
    """

    out_path: str = None
    summary_msg: str = field(default_factory=str)

    def print(self, msg: str) -> None:
        """
        Prints to system out and optionally to specified out_path.
        """
        print(msg)

        if self.out_path is not None:
            with open(self.out_path, "a+") as f:
                f.write(msg + "\n")

    @contextmanager
    def track(self, step):
        """
        Computes the elapsed time for given code context.
        """
        start = monotonic_ns()
        yield
        ns = monotonic_ns() - start
        msg = f"\n{'*' * 70}\n'{step}' took {ns / SEC_TO_NS_SCALE:.3f}s ({ns:,}ns)\n{'*' * 70}\n"
        print(msg)
        self.summary_msg += msg + "\n"

    def summary(self) -> None:
        """
        Prints summary of all benchmarks performed.
        """
        self.print(f"\n{'#' * 30}\nBenchmark Summary:\n{'#' * 30}\n\n{self.summary_msg}")


class BestRun(NamedTuple):
    """
    The best run found by a hyperparameter search (see [`~SetFitTrainer.hyperparameter_search`]).

    Parameters:
        run_id (`str`):
            The id of the best run.
        objective (`float`):
            The objective that was obtained for this run.
        hyperparameters (`Dict[str, Any]`):
            The hyperparameters picked to get this run.
        backend (`Any`):
            The relevant internal object used for optimization. For optuna this is the `study` object.
    """

    run_id: str
    objective: float
    hyperparameters: Dict[str, Any]
    backend: Any = None
