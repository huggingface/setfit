import numpy as np
import pytest

from setfit.sampler import ContrastiveDataset


@pytest.mark.parametrize(
    "sampling_strategy, expected_pos_pairs, expected_neg_pairs",
    [("unique", 1, 2), ("undersampling", 1, 1), ("oversampling", 2, 2)],
)
def test_sentence_pairs_generation(sampling_strategy: str, expected_pos_pairs: int, expected_neg_pairs: int):
    sentences = np.array(["sent 1", "sent 2", "sent 3"])
    labels = np.array(["label 1", "label 1", "label 2"])

    multilabel = False

    data_sampler = ContrastiveDataset(sentences, labels, multilabel, sampling_strategy=sampling_strategy)

    assert data_sampler.len_pos_pairs == expected_pos_pairs
    assert data_sampler.len_neg_pairs == expected_neg_pairs

    pairs = [i for i in data_sampler]

    assert len(pairs) == expected_pos_pairs + expected_neg_pairs
    assert pairs[0] == {"sentence_1": "sent 1", "sentence_2": "sent 2", "label": 1.0}


@pytest.mark.parametrize(
    "sampling_strategy, expected_pos_pairs, expected_neg_pairs",
    [("unique", 2, 4), ("undersampling", 2, 2), ("oversampling", 4, 4)],
)
def test_sentence_pairs_generation_multilabel(
    sampling_strategy: str, expected_pos_pairs: int, expected_neg_pairs: int
):
    sentences = np.array(["sent 1", "sent 2", "sent 3", "sent 4"])
    labels = np.array([[1, 0, 0, 1], [0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    multilabel = True

    data_sampler = ContrastiveDataset(sentences, labels, multilabel, sampling_strategy=sampling_strategy)
    assert data_sampler.len_pos_pairs == expected_pos_pairs
    assert data_sampler.len_neg_pairs == expected_neg_pairs

    pairs = [i for i in data_sampler]
    assert len(pairs) == expected_pos_pairs + expected_neg_pairs


@pytest.mark.parametrize("sampling_strategy", ["unique", "undersampling", "oversampling"])
@pytest.mark.parametrize("multilabel", [False, True])
def test_sentence_pairs_generation_excludes_identity_pairs(sampling_strategy: str, multilabel: bool):
    sentences = np.array(["sent 1", "sent 2", "sent 3", "sent 4"])
    if multilabel:
        labels = np.array([[1, 0, 0, 1], [0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    else:
        labels = np.array(["label 1", "label 1", "label 2", "label 2"])

    data_sampler = ContrastiveDataset(sentences, labels, multilabel, sampling_strategy=sampling_strategy)

    for pair in data_sampler:
        assert pair["sentence_1"] != pair["sentence_2"]


def test_sentence_pairs_generation_single_sentence_per_label():
    sentences = np.array(["sent 1", "sent 2"])
    labels = np.array(["label 1", "label 2"])

    data_sampler = ContrastiveDataset(sentences, labels, multilabel=False)

    assert data_sampler.len_pos_pairs == 0
    assert list(data_sampler) == [{"sentence_1": "sent 1", "sentence_2": "sent 2", "label": 0.0}]


def test_sentence_pairs_generation_single_label():
    sentences = np.array(["sent 1", "sent 2", "sent 3"])
    labels = np.array(["label 1", "label 1", "label 1"])

    data_sampler = ContrastiveDataset(sentences, labels, multilabel=False)

    assert data_sampler.len_neg_pairs == 0
    assert len(list(data_sampler)) == 3
