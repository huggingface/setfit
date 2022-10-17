import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier

from setfit import SetFitModel, SetFitHead
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


def test_setfit_model_body():
    model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-albert-small-v2")

    assert type(model.model_body) is SentenceTransformer


def test_setfit_default_model_head():
    model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-albert-small-v2")

    assert type(model.model_head) is LogisticRegression


def test_setfit_multilabel_one_vs_rest_model_head():
    model = SetFitModel.from_pretrained(
        "sentence-transformers/paraphrase-albert-small-v2", multi_target_strategy="one-vs-rest"
    )

    assert type(model.model_head) is OneVsRestClassifier


def test_setfit_multilabel_multi_output_classifier_model_head():
    model = SetFitModel.from_pretrained(
        "sentence-transformers/paraphrase-albert-small-v2", multi_target_strategy="multi-output"
    )

    assert type(model.model_head) is MultiOutputClassifier


def test_setfit_multilabel_classifier_chain_classifier_model_head():
    model = SetFitModel.from_pretrained(
        "sentence-transformers/paraphrase-albert-small-v2", multi_target_strategy="classifier-chain"
    )

    assert type(model.model_head) is ClassifierChain


def test_setfit_differentiable_head():
    model = SetFitModel.from_pretrained(
        "sentence-transformers/paraphrase-albert-small-v2", use_differentiable_head=True
    )

    assert type(model.model_head) is SetFitHead