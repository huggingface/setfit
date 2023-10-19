import pytest
import numpy as np

from sentence_transformers import InputExample

from setfit.sampler import ConstrastiveDataset


@pytest.mark.parametrize("sampling_strategy, expected_pos_pairs, expected_neg_pairs", [
    ("unique", 4, 2),
    ("undersampling", 2, 2),
    ("oversampling", 4, 4)
])
def test_sentence_pairs_generation(sampling_strategy: str, expected_pos_pairs: int, expected_neg_pairs: int):
    sentences = np.array(["sent 1", "sent 2", "sent 3"])
    labels = np.array(["label 1", "label 1", "label 2"])

    data = [InputExample(texts=[text], label=label) for text, label in zip(sentences, labels)]
    multilabel = False

    data_sampler = ConstrastiveDataset(data, multilabel, sampling_strategy=sampling_strategy)

    assert data_sampler.len_pos_pairs == expected_pos_pairs
    assert data_sampler.len_neg_pairs == expected_neg_pairs
    
    pairs = [i for i in data_sampler]

    assert len(pairs) == expected_pos_pairs + expected_neg_pairs
    assert pairs[0].texts == ["sent 1", "sent 1"]
    assert pairs[0].label == 1.0


@pytest.mark.parametrize("sampling_strategy, expected_pos_pairs, expected_neg_pairs", [
    ("unique", 6, 4),
    ("undersampling", 4, 4),
    ("oversampling", 6, 6)
])
def test_sentence_pairs_generation_multilabel(sampling_strategy: str, expected_pos_pairs: int, expected_neg_pairs: int):
    sentences = np.array(["sent 1", "sent 2", "sent 3", "sent 4"])
    labels = np.array([[1, 0, 0, 1], [0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    data = [InputExample(texts=[text], label=label) for text, label in zip(sentences, labels)]
    multilabel = True

    data_sampler = ConstrastiveDataset(data, multilabel, sampling_strategy=sampling_strategy)
    assert data_sampler.len_pos_pairs == expected_pos_pairs
    assert data_sampler.len_neg_pairs == expected_neg_pairs
    
    pairs = [i for i in data_sampler]
    assert len(pairs) == expected_pos_pairs + expected_neg_pairs
