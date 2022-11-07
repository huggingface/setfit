from typing import TYPE_CHECKING, Dict, List, Tuple

import pandas as pd
import torch
from datasets import Dataset, DatasetDict, concatenate_datasets
from torch.utils.data import Dataset as TorchDataset


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


TokenizerOutput = Dict[str, List[int]]
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
SAMPLE_SIZES = [2, 4, 8, 16, 32, 64]


def get_augmented_samples(dataset: str, sample_size: int = 2) -> Dict[str, list]:
    if dataset == "emotion":
        return {
            "text": ["The sentence is sadness"] * sample_size
            + ["The sentence is joy"] * sample_size
            + ["The sentence is love"] * sample_size
            + ["The sentence is anger"] * sample_size
            + ["The sentence is fear"] * sample_size
            + ["The sentence is surprise"] * sample_size,
            "label": [0] * sample_size
            + [1] * sample_size
            + [2] * sample_size
            + [3] * sample_size
            + [4] * sample_size
            + [5] * sample_size,
        }
    elif dataset == "ag_news":
        return {
            "text": ["The sentence is world"] * sample_size
            + ["The sentence is sports"] * sample_size
            + ["The sentence is business"] * sample_size
            + ["The sentence is tech"] * sample_size,
            "label": [0] * sample_size + [1] * sample_size + [2] * sample_size + [3] * sample_size,
        }
    elif dataset == "amazon_counterfactual_en":
        return {
            "text": ["The sentence is not counterfactual"] * sample_size
            + ["The sentence is counterfactual"] * sample_size,
            "label": [0] * sample_size + [1] * sample_size,
        }
    elif dataset == "SentEval-CR":
        return {
            "text": ["The sentence is negative"] * sample_size + ["The sentence is positive"] * sample_size,
            "label": [0] * sample_size + [1] * sample_size,
        }
    elif dataset == "sst5":
        return {
            "text": ["The sentence is very negative"] * sample_size
            + ["The sentence is negative"] * sample_size
            + ["The sentence is neutral"] * sample_size
            + ["The sentence is positive"] * sample_size
            + ["The sentence is very positive"] * sample_size,
            "label": [0] * sample_size + [1] * sample_size + [2] * sample_size + [3] * sample_size + [4] * sample_size,
        }
    elif dataset == "enron_spam":
        return {
            "text": ["The sentence is ham"] * sample_size + ["The sentence is spam"] * sample_size,
            "label": [0] * sample_size + [1] * sample_size,
        }
    elif dataset == "tweet_eval_stance_abortion":
        return {
            "text": ["The sentence is none"] * sample_size
            + ["The sentence is against"] * sample_size
            + ["The sentence is favor"] * sample_size,
            "label": [0] * sample_size + [1] * sample_size + [2] * sample_size,
        }
    elif dataset == "ade_corpus_v2_classification":
        return {
            "text": ["The sentence is not related"] * sample_size + ["The sentence is related"] * sample_size,
            "label": [0] * sample_size + [1] * sample_size,
        }
    else:
        raise ValueError(f"Dataset {dataset} not supported for data augmentation!")


def create_samples(df: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    """Samples a DataFrame to create an equal number of samples per class (when possible)."""
    examples = []
    for label in df["label"].unique():
        subset = df.query(f"label == {label}")
        if len(subset) > sample_size:
            examples.append(subset.sample(sample_size, random_state=seed, replace=False))
        else:
            examples.append(subset)
    return pd.concat(examples)


def sample_dataset(dataset: Dataset, label_column: str = "label", num_samples: int = 8, seed: int = 42) -> Dataset:
    """Samples a Dataset to create an equal number of samples per class (when possible)."""
    shuffled_dataset = dataset.shuffle(seed=seed)
    num_labels = len(dataset.unique(label_column))
    samples = []
    for label in range(num_labels):
        data = shuffled_dataset.filter(lambda example: int(example[label_column]) == label)
        num_samples = min(len(data) // 2, num_samples)
        samples.append(data.select([i for i in range(num_samples)]))

    all_samples = concatenate_datasets(samples)
    return all_samples.shuffle(seed=seed)


def create_fewshot_splits(
    dataset: Dataset, sample_sizes: List[int], add_data_augmentation: bool = False, dataset_name: str = None
) -> DatasetDict:
    """Creates training splits from the dataset with an equal number of samples per class (when possible)."""
    splits_ds = DatasetDict()
    df = dataset.to_pandas()

    for sample_size in sample_sizes:
        for idx, seed in enumerate(SEEDS):
            if add_data_augmentation and dataset_name is not None:
                augmented_samples = get_augmented_samples(dataset_name, sample_size)
                augmented_df = pd.DataFrame(augmented_samples)
                samples_df = create_samples(df, sample_size, seed)
                split_df = pd.concat([samples_df, augmented_df], axis=0).sample(frac=1, random_state=seed)
            else:
                split_df = create_samples(df, sample_size, seed)
            splits_ds[f"train-{sample_size}-{idx}"] = Dataset.from_pandas(split_df, preserve_index=False)
    return splits_ds


def create_samples_multilabel(df: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    """Samples a DataFrame to create an equal number of samples per class (when possible)."""
    examples = []
    column_labels = [_col for _col in df.columns.tolist() if _col != "text"]
    for label in column_labels:
        subset = df.query(f"{label} == 1")
        if len(subset) > sample_size:
            examples.append(subset.sample(sample_size, random_state=seed, replace=False))
        else:
            examples.append(subset)
    # Dropping duplicates for samples selected multiple times as they have multi labels
    return pd.concat(examples).drop_duplicates()


def create_fewshot_splits_multilabel(dataset: Dataset, sample_sizes: List[int]) -> DatasetDict:
    """Creates training splits from the dataset with an equal number of samples per class (when possible)."""
    splits_ds = DatasetDict()
    df = dataset.to_pandas()
    for sample_size in sample_sizes:
        for idx, seed in enumerate(SEEDS):
            split_df = create_samples_multilabel(df, sample_size, seed)
            splits_ds[f"train-{sample_size}-{idx}"] = Dataset.from_pandas(split_df, preserve_index=False)
    return splits_ds


def add_templated_examples(
    dataset: Dataset,
    candidate_labels: List[str],
    template: str = "This sentence is {}",
    sample_size: int = 2,
    text_column: str = "text",
    label_column: str = "label",
    multi_label: bool = False,
) -> Dataset:
    """Adds templated examples to a Dataset.

    The Dataset is assumed to have a text column with the name `text_column` and a
    label column with the name `label_column`, which contains one-hot or multi-hot
    encoded label sequences.

    Args:
        dataset (`Dataset`): The Dataset to add templated examples to.
        candidate_labels (`List[str]`): This list of candidate labels to be fed into
            the template to construct examples. This should align with the
            `label_column_name` column of `dataset`.
        template (`str`, *optional*, defaults to `"This sentence is {}"`): The template
            used to turn each label into a synthetic training example. This template
            must include a {} for the candidate label to be inserted into the template.
            For example, the default template is "This sentence is {}." With the
            candidate label "sports", this would produce an example
            "This sentence is sports".
        sample_size (`int`, *optional*, defaults to 2): The number of examples to
            make for each candidate label.
        text_column (`str`, *optional*, defaults to `"text"`): The name of the column
            containing the text of the examples.
        label_column (`str`, *optional*, defaults to `"label"`): The name of the column
            containing the labels of the examples.
        multi_label (`bool`, *optional*, defaults to `False`): Whether or not multiple candidate labels can be true.

    Returns:
        `Dataset`: A copy of the input Dataset with templated examples added.

    Raises:
        `ValueError`: If the input Dataset is not empty and one or both of the
            provided column names are missing.
    """
    required_columns = {text_column, label_column}
    column_names = set(dataset.column_names)
    if column_names:
        missing_columns = required_columns.difference(column_names)
        if missing_columns:
            raise ValueError(f"The following columns are missing from the input dataset: {missing_columns}.")

    empty_label_vector = [0] * len(candidate_labels)

    for label_id, label_name in enumerate(candidate_labels):
        label_vector = empty_label_vector.copy()
        label_vector[label_id] = 1
        example = {
            text_column: template.format(label_name),
            label_column: label_vector if multi_label else label_id,
        }
        for _ in range(sample_size):
            dataset = dataset.add_item(example)

    return dataset


class SetFitDataset(TorchDataset):
    """SetFitDataset

    A dataset for training the differentiable head on text classification.

    Args:
        x (`List[str]`):
            A list of input data as texts that will be fed into `SetFitModel`.
        y (`List[int]`):
            A list of input data's labels.
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer from `SetFitModel`'s body.
        max_length (`int`, defaults to `32`):
            The maximum token length a tokenizer can generate.
            Will pad or truncate tokens when the number of tokens for a text is either smaller or larger than this value.
    """

    def __init__(
        self,
        x: List[str],
        y: List[int],
        tokenizer: "PreTrainedTokenizerBase",
        max_length: int = 32,
    ) -> None:
        assert len(x) == len(y)

        self.x = x
        self.y = y
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[TokenizerOutput, int]:
        feature = self.tokenizer(
            self.x[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
        )
        label = self.y[idx]

        return feature, label

    @staticmethod
    def collate_fn(batch):
        features = {
            "input_ids": [],
            "attention_mask": [],
            "token_type_ids": [],
        }
        labels = []
        for feature, label in batch:
            features["input_ids"].append(feature["input_ids"])
            features["attention_mask"].append(feature["attention_mask"])
            features["token_type_ids"].append(feature["token_type_ids"])
            labels.append(label)

        # convert to tensors
        features = {k: torch.Tensor(v).int() for k, v in features.items()}
        labels = torch.Tensor(labels).long()

        return features, labels
