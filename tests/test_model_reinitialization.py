"""Offline regressions for embedding training after a hyperparameter model reset."""

import copy

import pytest
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer, models
from sklearn.linear_model import LogisticRegression
from transformers import BertConfig, BertModel, BertTokenizerFast
from transformers.trainer_utils import HPSearchBackend

from setfit import SetFitModel, Trainer, TrainingArguments


@pytest.fixture
def tiny_model(tmp_path):
    torch.manual_seed(42)
    vocab_path = tmp_path / "vocab.txt"
    vocab_path.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\ngood\nbad\nfilm\n", encoding="utf-8")
    BertTokenizerFast(vocab_file=str(vocab_path)).save_pretrained(tmp_path)
    config = BertConfig(
        vocab_size=8,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        hidden_dropout_prob=0,
        attention_probs_dropout_prob=0,
        max_position_embeddings=16,
    )
    BertModel(config).save_pretrained(tmp_path)
    body = SentenceTransformer(
        modules=[models.Transformer(str(tmp_path), max_seq_length=8), models.Pooling(8)], device="cpu"
    )
    return SetFitModel(body, LogisticRegression())


@pytest.mark.parametrize("train_before_reset", [False, True])
def test_hyperparameter_models_train_like_fresh_trainers(tiny_model, tmp_path, train_before_reset):
    dataset = Dataset.from_dict({"text": ["good film", "bad film", "film good", "film bad"], "label": [1, 0, 1, 0]})
    args = TrainingArguments(
        output_dir=str(tmp_path / "training"),
        batch_size=4,
        max_steps=3,
        num_epochs=1,
        warmup_proportion=0,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        seed=42,
    )
    trainer = Trainer(model_init=lambda params=None: copy.deepcopy(tiny_model), args=args, train_dataset=dataset)
    if train_before_reset:
        trainer.train()

    trainer.hp_search_backend = HPSearchBackend.OPTUNA
    for learning_rate in [0.01, 0.02]:
        old_body = trainer.model.model_body
        old_weights = copy.deepcopy(old_body.state_dict())
        before = copy.deepcopy(tiny_model.model_body.state_dict())
        trainer.train(trial={"body_learning_rate": learning_rate})

        assert any(
            not torch.equal(value, before[name]) for name, value in trainer.model.model_body.state_dict().items()
        )
        for name, value in old_body.state_dict().items():
            torch.testing.assert_close(value, old_weights[name], rtol=0, atol=0)
        assert trainer.st_trainer.model_wrapped is trainer.model.model_body
        optimized_parameters = {
            id(parameter) for group in trainer.st_trainer.optimizer.param_groups for parameter in group["params"]
        }
        assert optimized_parameters == {id(parameter) for parameter in trainer.model.model_body.parameters()}

        reference = Trainer(model=copy.deepcopy(tiny_model), args=copy.deepcopy(trainer.args), train_dataset=dataset)
        reference.train()
        for name, value in trainer.model.model_body.state_dict().items():
            torch.testing.assert_close(value, reference.model.model_body.state_dict()[name], rtol=0, atol=0)
        torch.testing.assert_close(
            trainer.model.predict_proba(dataset["text"], as_numpy=False),
            reference.model.predict_proba(dataset["text"], as_numpy=False),
        )
