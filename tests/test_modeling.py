import numpy as np
from sentence_transformers import InputExample
from setfit.modeling import sentence_pairs_generation, sentence_pairs_generation_multilabel

def test_sentence_pairs_generation():
    sentences = np.array(["sent 1", "sent 2", "sent 3"])
    labels = np.array(["label 1", "label 2", "label 3"])

    pairs = []
    n_iterations = 2

    for _ in range(n_iterations):
        pairs = sentence_pairs_generation(sentences, labels, pairs)

    assert len(pairs) == 12
    assert pairs[0].texts == ["sent 1", "sent 1"]
    assert pairs[0].label == 1.0



def test_sentence_pairs_generation_multilabel():
    sentences = np.array(["sent 1", "sent 2", "sent 3"])
    labels = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

    pairs = []
    n_iterations = 2

    for _ in range(n_iterations):
        pairs = sentence_pairs_generation_multilabel(sentences, labels, pairs)

    assert len(pairs) == 12
    assert pairs[0].texts == ["sent 1", "sent 1"]
    assert pairs[0].label == 1.0
