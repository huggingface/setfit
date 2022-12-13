import string

import pandas as pd
import pytest
from datasets import Dataset

from setfit.data import (
    SAMPLE_SIZES,
    SEEDS,
    add_templated_examples,
    create_fewshot_splits,
    create_samples,
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
    augmented_dataset = add_templated_examples(
        empty_dataset, candidate_labels=["label-0", "label-1"], multi_label=True
    )

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
    augmented_dataset = add_templated_examples(dataset, candidate_labels=["label-0", "label-1"], multi_label=True)

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
        add_templated_examples(
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
    splits_ds = create_fewshot_splits(dataset, SAMPLE_SIZES)
    assert len(splits_ds) == len(SAMPLE_SIZES) * len(SEEDS)


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
    # The dataset ought to have just `num_samples` rows, as `unbalanced_dataset`
    # only has one label for which multiple entries exist. So, we can only sample
    # for one label, leading to `num_samples` rows in the final dataset.
    assert ds.num_rows == num_samples
