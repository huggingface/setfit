from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import Dataset as TorchDataset

from . import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


TokenizerOutput = Dict[str, List[int]]
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
SAMPLE_SIZES = [2, 4, 8, 16, 32, 64]


def get_templated_dataset(
    dataset: Optional[Dataset] = None,
    candidate_labels: Optional[List[str]] = None,
    reference_dataset: Optional[str] = None,
    template: str = "This sentence is {}",
    sample_size: int = 2,
    text_column: str = "text",
    label_column: str = "label",
    multi_label: bool = False,
    label_names_column: str = "label_text",
) -> Dataset:
    """Create templated examples for a reference dataset or reference labels.

    If `candidate_labels` is supplied, use it for generating the templates.
    Otherwise, use the labels loaded from `reference_dataset`.

    If input Dataset is supplied, add the examples to it, otherwise create a new Dataset.
    The input Dataset is assumed to have a text column with the name `text_column` and a
    label column with the name `label_column`, which contains one-hot or multi-hot
    encoded label sequences.

    Args:
        dataset (`Dataset`, *optional*): A Dataset to add templated examples to.
        candidate_labels (`List[str]`, *optional*): The list of candidate
            labels to be fed into the template to construct examples.
        reference_dataset (`str`, *optional*): A dataset to take labels
            from, if `candidate_labels` is not supplied.
        template (`str`, *optional*, defaults to `"This sentence is {}"`): The template
            used to turn each label into a synthetic training example. This template
            must include a {} for the candidate label to be inserted into the template.
            For example, the default template is "This sentence is {}." With the
            candidate label "sports", this would produce an example
            "This sentence is sports".
        sample_size (`int`, *optional*, defaults to 2): The number of examples to make for
            each candidate label.
        text_column (`str`, *optional*, defaults to `"text"`): The name of the column
            containing the text of the examples.
        label_column (`str`, *optional*, defaults to `"label"`): The name of the column
            in `dataset` containing the labels of the examples.
        multi_label (`bool`, *optional*, defaults to `False`): Whether or not multiple
            candidate labels can be true.
        label_names_column (`str`, *optional*, defaults to "label_text"): The name of the
            label column in the `reference_dataset`, to be used in case there is no ClassLabel
            feature for the label column.

    Returns:
        `Dataset`: A copy of the input Dataset with templated examples added.

    Raises:
        `ValueError`: If the input Dataset is not empty and one or both of the
            provided column names are missing.
    """
    if dataset is None:
        dataset = Dataset.from_dict({})

    required_columns = {text_column, label_column}
    column_names = set(dataset.column_names)
    if column_names:
        missing_columns = required_columns.difference(column_names)
        if missing_columns:
            raise ValueError(f"The following columns are missing from the input dataset: {missing_columns}.")

    if bool(reference_dataset) == bool(candidate_labels):
        raise ValueError(
            "Must supply exactly one of `reference_dataset` or `candidate_labels` to `get_templated_dataset()`!"
        )

    if candidate_labels is None:
        candidate_labels = get_candidate_labels(reference_dataset, label_names_column)

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


def get_candidate_labels(dataset_name: str, label_names_column: str = "label_text") -> List[str]:
    dataset = load_dataset(dataset_name, split="train")

    try:
        # Extract ClassLabel feature from "label" column
        label_features = dataset.features["label"]
        # Label names to classify with
        candidate_labels = label_features.names

    except AttributeError:
        # Some datasets on the Hugging Face Hub don't have a ClassLabel feature for the label column.
        # In these cases, you should compute the candidate labels manually by first computing the id2label mapping.

        # The column with the label names
        label_names = dataset.unique(label_names_column)
        # The column with the label IDs
        label_ids = dataset.unique("label")

        # Compute the id2label mapping and sort by label ID
        id2label = sorted(zip(label_ids, label_names), key=lambda x: x[0])

        candidate_labels = list(map(lambda x: x[1], id2label))

    return candidate_labels


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

    df = shuffled_dataset.to_pandas()
    df = df.groupby(label_column)

    # sample num_samples, or at least as much as possible
    df = df.apply(lambda x: x.sample(min(num_samples, len(x)), random_state=seed))
    df = df.reset_index(drop=True)

    all_samples = Dataset.from_pandas(df, features=dataset.features)
    return all_samples.shuffle(seed=seed)


def create_fewshot_splits(
    dataset: Dataset,
    sample_sizes: List[int],
    add_data_augmentation: bool = False,
    dataset_name: Optional[str] = None,
) -> DatasetDict:
    """Creates training splits from the dataset with an equal number of samples per class (when possible)."""
    splits_ds = DatasetDict()
    df = dataset.to_pandas()

    if add_data_augmentation and dataset_name is None:
        raise ValueError(
            "If `add_data_augmentation` is True, must supply a `dataset_name` to create_fewshot_splits()!"
        )

    for sample_size in sample_sizes:
        if add_data_augmentation:
            augmented_df = get_templated_dataset(reference_dataset=dataset_name, sample_size=sample_size).to_pandas()
        for idx, seed in enumerate(SEEDS):
            split_df = create_samples(df, sample_size, seed)
            if add_data_augmentation:
                split_df = pd.concat([split_df, augmented_df], axis=0).sample(frac=1, random_state=seed)
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


class SetFitDataset(TorchDataset):
    """SetFitDataset

    A dataset for training the differentiable head on text classification.

    Args:
        x (`List[str]`):
            A list of input data as texts that will be fed into `SetFitModel`.
        y (`Union[List[int], List[List[int]]]`):
            A list of input data's labels. Can be a nested list for multi-label classification.
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer from `SetFitModel`'s body.
        max_length (`int`, defaults to `32`):
            The maximum token length a tokenizer can generate.
            Will pad or truncate tokens when the number of tokens for a text is either smaller or larger than this value.
    """

    def __init__(
        self,
        x: List[str],
        y: Union[List[int], List[List[int]]],
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

    def __getitem__(self, idx: int) -> Tuple[TokenizerOutput, Union[int, List[int]]]:
        feature = self.tokenizer(
            self.x[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask="attention_mask" in self.tokenizer.model_input_names,
            return_token_type_ids="token_type_ids" in self.tokenizer.model_input_names,
        )
        label = self.y[idx]

        return feature, label

    def collate_fn(self, batch):
        features = {input_name: [] for input_name in self.tokenizer.model_input_names}

        labels = []
        for feature, label in batch:
            features["input_ids"].append(feature["input_ids"])
            if "attention_mask" in features:
                features["attention_mask"].append(feature["attention_mask"])
            if "token_type_ids" in features:
                features["token_type_ids"].append(feature["token_type_ids"])
            labels.append(label)

        # convert to tensors
        features = {k: torch.Tensor(v).int() for k, v in features.items()}
        labels = torch.Tensor(labels)
        labels = labels.long() if len(labels.size()) == 1 else labels.float()
        return features, labels
