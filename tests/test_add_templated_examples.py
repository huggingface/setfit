import pytest
from datasets import Dataset

from setfit.data import add_templated_examples


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
