from contextlib import contextmanager
from dataclasses import dataclass, field
from time import monotonic_ns
from typing import Any, Dict, List, NamedTuple, Tuple

from datasets import Dataset, DatasetDict, load_dataset
from sentence_transformers import losses

from .data import create_fewshot_splits, create_fewshot_splits_multilabel
from .modeling import SupConLoss


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


def load_data_splits(
    dataset: str, sample_sizes: List[int], add_data_augmentation: bool = False
) -> Tuple[DatasetDict, Dataset]:
    """Loads a dataset from the Hugging Face Hub and returns the test split and few-shot training splits."""
    print(f"\n\n\n============== {dataset} ============")
    # Load one of the SetFit training sets from the Hugging Face Hub
    train_split = load_dataset(f"SetFit/{dataset}", split="train")
    train_splits = create_fewshot_splits(train_split, sample_sizes, add_data_augmentation, dataset)
    test_split = load_dataset(f"SetFit/{dataset}", split="test")
    print(f"Test set: {len(test_split)}")
    return train_splits, test_split


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
