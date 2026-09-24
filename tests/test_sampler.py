import numpy as np
import pytest

from setfit.sampler import ContrastiveDataset


@pytest.mark.parametrize(
    "sampling_strategy, expected_pos_pairs, expected_neg_pairs",
    [("unique", 4, 2), ("undersampling", 2, 2), ("oversampling", 4, 4)],
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
    assert pairs[0] == {"sentence_1": "sent 1", "sentence_2": "sent 1", "label": 1.0}


@pytest.mark.parametrize(
    "sampling_strategy, expected_pos_pairs, expected_neg_pairs",
    [("unique", 6, 4), ("undersampling", 4, 4), ("oversampling", 6, 6)],
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


def test_num_iterations_bounds_pair_generation() -> None:
    """With ``num_iterations`` only the first ``num_iterations * n`` pairs per bucket are served, so the sampler must
    not enumerate all n^2 candidates: 3,000 inputs would otherwise mean 4.5M pair dicts before training starts."""
    import time

    n, num_iterations = 3000, 5
    started = time.perf_counter()
    dataset = ContrastiveDataset([f"text {i}" for i in range(n)], [i % 6 for i in range(n)], False, num_iterations)
    assert len(dataset) == 2 * num_iterations * n
    assert len(dataset.pos_pairs) <= num_iterations * n
    assert len(dataset.neg_pairs) <= num_iterations * n
    assert len(list(dataset)) == 2 * num_iterations * n
    assert time.perf_counter() - started < 10
