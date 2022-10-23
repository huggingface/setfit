from unittest import TestCase

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier

from setfit import SetFitHead, SetFitModel
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


class SetFitModelDifferentiableHeadTest(TestCase):
    @classmethod
    def setUpClass(cls):
        dataset = load_dataset("sst2")
        num_classes = 2
        train_dataset = dataset["train"].shuffle(seed=42).select(range(2 * num_classes))
        x_train, y_train = train_dataset["sentence"], train_dataset["label"]

        model = cls._build_model(num_classes)
        model.unfreeze()  # unfreeze the model body and head

        # run one step
        model.model_body.train()
        model.model_head.train()

        dataloader = model._prepare_dataloader(x_train, y_train, batch_size=2 * num_classes)
        criterion = model.model_head.get_loss_fn()
        optimizer = model._prepare_optimizer(2e-4, None, 0.1)

        batch = next(iter(dataloader))
        features, labels = batch
        optimizer.zero_grad()

        outputs = model.model_body(features)
        outputs = model.model_head(outputs)
        loss = criterion(outputs["prediction"], labels)
        loss.backward()
        optimizer.step()

        cls.model = model
        cls.out_features = num_classes

    @staticmethod
    def _build_model(num_classes: int) -> SetFitModel:
        model = SetFitModel.from_pretrained(
            "sentence-transformers/paraphrase-albert-small-v2",
            use_differentiable_head=True,
            head_params={"out_features": num_classes},
        )

        return model

    def test_setfit_single_target_differentiable_head(self):
        model = self._build_model(num_classes=1)

        assert type(model.model_head) is SetFitHead
        assert model.model_head.out_features == 1

    def test_setfit_multi_targets_differentiable_head(self):
        assert type(self.model.model_head) is SetFitHead
        assert self.model.model_head.out_features == self.out_features

    def test_setfit_model_forward(self):
        # Already ran the model's forward in the fixture, so do simple testing here.
        assert type(self.model) is SetFitModel

    def test_setfit_model_backward(self):
        # check the model head's gradients
        for name, param in self.model.model_head.named_parameters():
            assert param.grad is not None, f"Gradients of {name} in the model head is None."
            assert not (param.grad == 0).all().item(), f"All gradients of {name} in the model head are zeros."
            assert not param.grad.isnan().any().item(), f"Gradients of {name} in the model head have NaN."
            assert not param.grad.isinf().any().item(), f"Gradients of {name} in the model head have Inf."

        # check the model body's gradients
        for name, param in self.model.model_body.named_parameters():
            if "0.auto_model.pooler" in name:  # ignore pooler
                continue

            assert param.grad is not None, f"Gradients of {name} in the model body is None."
            assert not (param.grad == 0).all().item(), f"All gradients of {name} in the model body are zeros."
            assert not param.grad.isnan().any().item(), f"Gradients of {name} in the model body have NaN."
            assert not param.grad.isinf().any().item(), f"Gradients of {name} in the model body have Inf."
