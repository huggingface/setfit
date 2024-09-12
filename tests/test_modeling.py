import json
from pathlib import Path
from unittest import TestCase

import numpy as np
import pytest
import torch
from datasets import Dataset, load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier

from setfit import SetFitHead, SetFitModel, Trainer
from setfit.modeling import MODEL_HEAD_NAME
from tests.utils import SafeTemporaryDirectory


torch_cuda_available = pytest.mark.skipif(not torch.cuda.is_available(), reason="PyTorch must be compiled with CUDA")


def test_setfit_model_body():
    model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-albert-small-v2")

    assert type(model.model_body) is SentenceTransformer


def test_setfit_default_model_head():
    model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-albert-small-v2")

    assert type(model.model_head) is LogisticRegression


def test_setfit_model_head_params():
    params = {
        "head_params": {
            "max_iter": 200,
            "solver": "newton-cg",
        }
    }

    model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-albert-small-v2", **params)

    assert type(model.model_head) is LogisticRegression
    assert params["head_params"] == {
        parameter: value
        for parameter, value in model.model_head.get_params(deep=False).items()
        if parameter in params["head_params"]
    }


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
        device = model.model_head.device

        batch = next(iter(dataloader))
        features, labels = batch
        features = {k: v.to(device) for k, v in features.items()}
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = model.model_body(features)
        outputs = model.model_head(outputs)
        loss = criterion(outputs["logits"], labels)
        loss.backward()
        optimizer.step()

        cls.model = model
        cls.out_features = num_classes
        cls.x_train = x_train
        cls.y_train = y_train

    @staticmethod
    def _build_model(num_classes: int) -> SetFitModel:
        model = SetFitModel.from_pretrained(
            "sentence-transformers/paraphrase-albert-small-v2",
            use_differentiable_head=True,
            head_params={"out_features": num_classes},
        )

        return model

    def test_setfit_body_and_head_on_same_device(self):
        model = self._build_model(num_classes=1)
        assert model.model_body.device.type == model.model_head.device.type

    def test_setfit_single_target_differentiable_head(self):
        model = self._build_model(num_classes=1)

        assert type(model.model_head) is SetFitHead
        assert model.model_head.out_features == 2

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

    def test_max_length_is_larger_than_max_acceptable_length(self):
        max_length = int(1e6)
        dataloader = self.model._prepare_dataloader(self.x_train, self.y_train, batch_size=1, max_length=max_length)

        assert dataloader.dataset.max_length == self.model.model_body.get_max_seq_length()

    def test_max_length_is_smaller_than_max_acceptable_length(self):
        max_length = 32
        dataloader = self.model._prepare_dataloader(self.x_train, self.y_train, batch_size=1, max_length=max_length)

        assert dataloader.dataset.max_length == max_length


def test_setfit_from_pretrained_local_model_without_head(tmp_path):
    model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-albert-small-v2")
    model.save_pretrained(str(tmp_path.absolute()))

    (tmp_path / MODEL_HEAD_NAME).unlink()  # Delete head

    model = SetFitModel.from_pretrained(str(tmp_path.absolute()))

    assert isinstance(model, SetFitModel)


def test_setfit_from_pretrained_local_model_with_head(tmp_path):
    model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-albert-small-v2")
    model.save_pretrained(str(tmp_path.absolute()))

    model = SetFitModel.from_pretrained(str(tmp_path.absolute()))

    assert isinstance(model, SetFitModel)


def test_setfithead_multitarget_from_pretrained():
    model = SetFitModel.from_pretrained(
        "sentence-transformers/paraphrase-albert-small-v2",
        use_differentiable_head=True,
        multi_target_strategy="one-vs-rest",
        head_params={"out_features": 5},
    )
    assert isinstance(model.model_head, SetFitHead)
    assert model.model_head.multitarget
    assert isinstance(model.model_head.get_loss_fn(), torch.nn.BCEWithLogitsLoss)

    y_pred = model.predict("Test text")
    assert len(y_pred) == 5

    y_pred_probs = model.predict_proba("Test text", as_numpy=True)
    assert not np.isclose(y_pred_probs.sum(), 1)  # Should not sum to one


def test_to_logistic_head():
    model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-albert-small-v2")
    devices = (
        [torch.device("cpu"), torch.device("cuda", 0), torch.device("cpu")]
        if torch.cuda.is_available()
        else [torch.device("cpu")]
    )
    for device in devices:
        model.to(device)
        assert model.model_body.device == device


def test_to_torch_head():
    model = SetFitModel.from_pretrained(
        "sentence-transformers/paraphrase-albert-small-v2", use_differentiable_head=True
    )
    devices = (
        [torch.device("cpu"), torch.device("cuda", 0), torch.device("cpu")]
        if torch.cuda.is_available()
        else [torch.device("cpu")]
    )
    for device in devices:
        model.to(device)
        assert model.model_body.device == device
        assert model.model_head.device == device


@torch_cuda_available
@pytest.mark.parametrize("use_differentiable_head", [True, False])
def test_to_sentence_transformer_device_reset(use_differentiable_head):
    # This should initialize SentenceTransformer() without a specific device
    # which sets the model to CUDA iff CUDA is available.
    model = SetFitModel.from_pretrained(
        "sentence-transformers/paraphrase-albert-small-v2", use_differentiable_head=use_differentiable_head
    )
    # If we move the entire model to CPU, we expect it to stay on CPU forever,
    # Even after encoding or fitting
    model.to("cpu")
    assert model.model_body.device == torch.device("cpu")

    model.model_body.encode("This is a test sample to encode")
    assert model.model_body.device == torch.device("cpu")


@torch_cuda_available
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_load_model_on_device(device):
    model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-albert-small-v2", device=device)
    assert model.device.type == device
    assert model.model_body.device.type == device

    model.model_body.encode("This is a test sample to encode")


def test_save_load_config(model: SetFitModel) -> None:
    with SafeTemporaryDirectory() as tmp_dir:
        tmp_dir = str(Path(tmp_dir) / "model")
        model.save_pretrained(tmp_dir)
        config_path = Path(tmp_dir) / "config_setfit.json"
        assert config_path.exists()
        with open(config_path, "r") as f:
            config = json.load(f)
        assert config == {"normalize_embeddings": False, "labels": None}

    with SafeTemporaryDirectory() as tmp_dir:
        tmp_dir = str(Path(tmp_dir) / "model")
        model.normalize_embeddings = True
        model.labels = ["negative", "positive"]
        model.save_pretrained(tmp_dir)
        config_path = Path(tmp_dir) / "config_setfit.json"
        assert config_path.exists()
        with open(config_path, "r") as f:
            config = json.load(f)
        assert config == {"normalize_embeddings": True, "labels": ["negative", "positive"]}

        fresh_model = model.from_pretrained(tmp_dir)
        assert fresh_model.normalize_embeddings is True
        assert fresh_model.labels == ["negative", "positive"]


def test_load_model() -> None:
    model = SetFitModel.from_pretrained(
        "sentence-transformers/paraphrase-albert-small-v2", labels=["foo", "bar", "baz"]
    )
    assert model.labels == ["foo", "bar", "baz"]
    assert model.label2id == {"foo": 0, "bar": 1, "baz": 2}
    assert model.id2label == {0: "foo", 1: "bar", 2: "baz"}


def test_inference_with_labels() -> None:
    model = SetFitModel.from_pretrained("SetFit/test-setfit-sst2")
    assert model.labels is None
    assert model.predict(["Very good"]) == torch.tensor([1], dtype=torch.int32)
    model.labels = ["negative", "positive"]
    assert model.predict(["Very good"]) == ["positive"]

    model = SetFitModel.from_pretrained("SetFit/test-setfit-sst2-string-labels")
    assert model.labels is None
    assert model.predict(["Very good"]) == np.array(["positive"], dtype="<U8")
    model.labels = ["negative", "positive"]
    assert model.predict(["Very good"]) == ["positive"]

    model = SetFitModel.from_pretrained("SetFit/test-setfit-sst2-diff-head")
    assert model.labels is None
    assert model.predict(["Very good"]) == torch.tensor([1], dtype=torch.int32, device=model.device)
    model.labels = ["negative", "positive"]
    assert model.predict(["Very good"]) == ["positive"]


def test_singular_predict() -> None:
    model = SetFitModel.from_pretrained("SetFit/test-setfit-sst2")
    assert model.predict("That was cool!") == torch.tensor(1, dtype=torch.int32)
    probs = model.predict_proba("That was cool!")
    assert probs.shape == (2,)
    assert probs.argmax() == 1
    model.labels = ["negative", "positive"]
    assert model("That was cool!") == "positive"


# A differentiable head may still cause unexpected performance
@pytest.mark.parametrize("use_differentiable_head", [False])
def test_predict_proba_multi_output(use_differentiable_head: bool) -> None:
    model = SetFitModel.from_pretrained(
        "sentence-transformers/paraphrase-albert-small-v2",
        multi_target_strategy="multi-output",
        use_differentiable_head=use_differentiable_head,
    )
    train_dataset = Dataset.from_dict({"text": ["Hello", "World"], "label": [[1, 0], [0, 1]]})

    trainer = Trainer(model=model, train_dataset=train_dataset)
    trainer.train()

    outputs = model.predict_proba("That was cool!")
    assert isinstance(outputs, torch.Tensor)
    assert outputs.shape == (2, 2)
    outputs = model.predict_proba("That was cool!", as_numpy=True)
    assert isinstance(outputs, np.ndarray)
    assert outputs.shape == (2, 2)

    outputs = model.predict_proba(["That was cool!"] * 3)
    assert isinstance(outputs, torch.Tensor)
    assert outputs.shape == (3, 2, 2)
    outputs = model.predict_proba(["That was cool!"] * 3, as_numpy=True)
    assert isinstance(outputs, np.ndarray)
    assert outputs.shape == (3, 2, 2)
