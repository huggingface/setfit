from typing import List, Tuple

from datasets import Dataset, DatasetDict, load_dataset

from setfit.data import create_fewshot_splits

from dataclasses import dataclass, field
from time import monotonic_ns
from dataclasses import dataclass, field
from contextlib import contextmanager

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
        self.summary_msg += msg + '\n'

    def summary(self) -> None:
        """
        Prints summary of all benchmarks performed.
        """
        self.print(f"\n{'#' * 30}\nBenchmark Summary:\n{'#' * 30}\n\n{self.summary_msg}")
