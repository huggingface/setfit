import string

import numpy as np
import pandas as pd
import pytest
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from setfit.data import (
    SAMPLE_SIZES,
    SEEDS,
    SetFitDataset,
    create_fewshot_splits,
    create_fewshot_splits_multilabel,
    create_samples,
    get_templated_dataset,
    sample_dataset,
)


@pytest.fixture
def empty_dataset():
    return Dataset.from_dict({})


@pytest.fixture
def dataset():
    return Dataset.from_dict(
        {
            "text": ["label-0 text", "label-1 text"],
            "label": [[1, 0], [0, 1]],
        }
    )


@pytest.fixture
def unbalanced_dataset():
    return Dataset.from_dict({"text": string.ascii_letters, "label": [0] + 51 * [1]})


def test_add_to_empty_dataset_defaults(empty_dataset):
    augmented_dataset = get_templated_dataset(empty_dataset, candidate_labels=["label-0", "label-1"], multi_label=True)

    assert augmented_dataset[:] == {
        "text": [
            "This sentence is label-0",
            "This sentence is label-0",
            "This sentence is label-1",
            "This sentence is label-1",
        ],
        "label": [[1, 0], [1, 0], [0, 1], [0, 1]],
    }


def test_add_to_dataset_defaults(dataset):
    augmented_dataset = get_templated_dataset(dataset, candidate_labels=["label-0", "label-1"], multi_label=True)

    assert augmented_dataset[:] == {
        "text": [
            "label-0 text",
            "label-1 text",
            "This sentence is label-0",
            "This sentence is label-0",
            "This sentence is label-1",
            "This sentence is label-1",
        ],
        "label": [[1, 0], [0, 1], [1, 0], [1, 0], [0, 1], [0, 1]],
    }


@pytest.mark.parametrize(
    "text_column, label_column",
    [
        ("missing-text", "label"),
        ("text", "missing-label"),
        ("missing-text", "missing-label"),
    ],
)
def test_missing_columns(dataset, text_column, label_column):
    with pytest.raises(ValueError):
        get_templated_dataset(
            dataset,
            candidate_labels=["label-0", "label-1"],
            text_column=text_column,
            label_column=label_column,
        )


@pytest.mark.parametrize("sample_size", [8, 16, 32])
def test_subset_is_larger_than_sample_size(sample_size):
    data = {"label": [0] * 50 + [1] * 50}
    df = pd.DataFrame(data)
    sample_df = create_samples(df, sample_size=sample_size, seed=0)
    assert len(sample_df) == (sample_size * 2)


@pytest.mark.parametrize("sample_size", [8, 16, 32])
def test_subset_is_smaller_than_sample_size(sample_size):
    data = {"label": [0] * 3 + [1] * 3}
    df = pd.DataFrame(data)
    sample_df = create_samples(df, sample_size=sample_size, seed=0)
    assert len(sample_df) == len(df)


def test_expected_number_of_splits():
    dataset = Dataset.from_pandas(pd.DataFrame({"label": [0] * 50 + [1] * 50}))
    num_labels = 2
    splits_ds = create_fewshot_splits(dataset, SAMPLE_SIZES)
    assert len(splits_ds) == len(SAMPLE_SIZES) * len(SEEDS)

    split: Dataset
    for idx, split in enumerate(splits_ds.values()):
        sample_size = SAMPLE_SIZES[idx // len(SEEDS)]
        # The number of rows is limited by 100 due to the size of the original dataset
        assert len(split) == min(sample_size * num_labels, len(dataset))


def test_create_fewshot_splits_with_augmentation():
    dataset_name = "sst5"
    dataset = load_dataset(f"SetFit/{dataset_name}", split="train")
    num_labels = len(set(dataset["label"]))
    splits_ds = create_fewshot_splits(
        dataset, SAMPLE_SIZES, add_data_augmentation=True, dataset_name=f"SetFit/{dataset_name}"
    )
    assert len(splits_ds) == len(SAMPLE_SIZES) * len(SEEDS)

    split: Dataset
    for idx, split in enumerate(splits_ds.values()):
        sample_size = SAMPLE_SIZES[idx // len(SEEDS)]
        # Each split should have sample_size * num_labels * 2 rows:
        # for each label we sample `sample_size`, and then we generate
        # another `sample_size` samples through augmentation.
        assert len(split) == sample_size * num_labels * 2


def test_create_fewshot_splits_multilabel():
    num_samples = 50
    dataset = Dataset.from_dict(
        {
            "text": string.ascii_letters[:50],
            "label_one": np.random.randint(2, size=(num_samples,)),
            "label_two": np.random.randint(2, size=(num_samples,)),
            "label_three": np.random.randint(2, size=(num_samples,)),
        }
    )
    splits_ds = create_fewshot_splits_multilabel(dataset, SAMPLE_SIZES)
    assert len(splits_ds) == len(SAMPLE_SIZES) * len(SEEDS)
    # We can't safely test the number of rows of each of the splits
    # as duplicate samples are removed.


def test_sample_dataset_returns_expected_samples():
    num_samples = 2
    dataset = Dataset.from_dict({"text": ["hello"] * 50, "label": [0] * 25 + [1] * 25})
    samples = sample_dataset(dataset=dataset, num_samples=num_samples)
    for label_id in range(num_samples):
        assert len(samples.filter(lambda x: x["label"] == label_id)) == num_samples


def test_sample_dataset_with_label_column():
    num_samples = 2
    label_column = "my_labels"
    dataset = Dataset.from_dict({"text": ["hello"] * 50, label_column: [0] * 25 + [1] * 25})
    samples = sample_dataset(dataset=dataset, label_column=label_column, num_samples=num_samples)
    for label_id in range(num_samples):
        assert len(samples.filter(lambda x: x[label_column] == label_id)) == num_samples


def test_sample_dataset_with_unbalanced_ds(unbalanced_dataset):
    num_samples = 8
    ds = sample_dataset(unbalanced_dataset, num_samples=num_samples)
    # The dataset ought to have just `num_samples + 1` rows, as `unbalanced_dataset`
    # has one label with more than `num_samples` entries and another label with just 1 row.
    # We sample `num_samples` from the former, and 1 from the latter.
    assert ds.num_rows == num_samples + 1


@pytest.mark.parametrize(
    "dataset",
    [
        "SetFit/emotion",
        "SetFit/ag_news",
        "SetFit/amazon_counterfactual_en",
        "SetFit/SentEval-CR",
        "SetFit/sst5",
        "SetFit/enron_spam",
        "SetFit/tweet_eval_stance_abortion",
        "SetFit/ade_corpus_v2_classification",
    ],
)
def test_get_augmented_samples(dataset: str):
    dataset = get_templated_dataset(reference_dataset=dataset)
    assert set(dataset.column_names) == {"text", "label"}
    assert len(dataset["text"])
    assert len(dataset["label"])


def test_get_augmented_samples_negative():
    with pytest.raises(ValueError):
        get_templated_dataset(reference_dataset=None, candidate_labels=None)


@pytest.mark.parametrize(
    "tokenizer_name",
    ["sentence-transformers/paraphrase-albert-small-v2", "sentence-transformers/distiluse-base-multilingual-cased-v1"],
)
def test_correct_model_inputs(tokenizer_name):
    # Arbitrary testing data
    x = list(string.ascii_lowercase)
    y = list(range(len(x)))

    # Relatively Standard DataLoader setup using a SetFitDataset
    # for training a differentiable classification head
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    dataset = SetFitDataset(x, y, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=dataset.collate_fn,
        shuffle=True,
        pin_memory=True,
    )

    # Verify that the x_batch contains exactly those keys that the model requires
    x_batch, _ = next(iter(dataloader))
    assert set(x_batch.keys()) == set(tokenizer.model_input_names)


def test_preserve_features() -> None:
    dataset = load_dataset("SetFit/sst5", split="train[:100]")
    label_column = "label_text"
    dataset = dataset.class_encode_column(label_column)
    train_dataset = sample_dataset(dataset, label_column=label_column, num_samples=8)
    assert train_dataset.features[label_column] == dataset.features[label_column]
