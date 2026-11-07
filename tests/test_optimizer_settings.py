"""Offline optimizer regressions using a small, randomly initialized encoder."""

import copy

import pytest
import torch
from sentence_transformers import SentenceTransformer, models
from transformers import BertConfig, BertModel, BertTokenizerFast

from setfit import SetFitHead, SetFitModel


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
    return SetFitModel(body, SetFitHead(in_features=8, out_features=2, device="cpu"))


@pytest.mark.parametrize("multitarget", [False, True])
@pytest.mark.parametrize("head_learning_rate", [0.0, 0.1])
def test_fit_preserves_body_with_zero_learning_rate(tiny_model, multitarget, head_learning_rate):
    tiny_model.model_head.multitarget = multitarget
    body_before = copy.deepcopy(tiny_model.model_body.state_dict())
    head_before = copy.deepcopy(tiny_model.model_head.state_dict())
    tiny_model.fit(
        ["good film", "bad film"],
        [[0, 1], [1, 0]] if multitarget else [1, 0],
        num_epochs=1,
        batch_size=2,
        body_learning_rate=0.0,
        head_learning_rate=head_learning_rate,
        l2_weight=0.0,
        end_to_end=True,
        max_length=8,
        show_progress_bar=False,
    )
    for name, value in tiny_model.model_body.state_dict().items():
        torch.testing.assert_close(value, body_before[name], rtol=0, atol=0)
    head_changed = any(
        not torch.equal(value, head_before[name]) for name, value in tiny_model.model_head.state_dict().items()
    )
    assert head_changed == (head_learning_rate != 0)


@pytest.mark.parametrize("multitarget", [False, True])
@pytest.mark.parametrize("l2_weight", [0.0, 0.2, None])
def test_fit_matches_adamw_weight_decay(tiny_model, l2_weight, multitarget):
    tiny_model.model_head.multitarget = multitarget
    targets = [[0, 1], [1, 0]] if multitarget else [1, 0]
    reference = copy.deepcopy(tiny_model)
    reference.model_body.train()
    reference.model_head.train()
    features, labels = next(
        iter(
            reference._prepare_dataloader(
                ["good film", "bad film"], targets, batch_size=2, max_length=8, shuffle=False
            )
        )
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": reference.model_body.parameters(), "lr": 0.01},
            {"params": reference.model_head.parameters(), "lr": 0.1},
        ],
        weight_decay=0.01 if l2_weight is None else l2_weight,
    )
    logits = reference.model_head(reference.model_body(features))["logits"]
    reference.model_head.get_loss_fn()(logits, labels).backward()
    optimizer.step()

    tiny_model.fit(
        ["good film", "bad film"],
        targets,
        num_epochs=1,
        batch_size=2,
        body_learning_rate=0.01,
        head_learning_rate=0.1,
        l2_weight=l2_weight,
        end_to_end=True,
        max_length=8,
        show_progress_bar=False,
    )
    for actual, expected in [
        (tiny_model.model_body, reference.model_body),
        (tiny_model.model_head, reference.model_head),
    ]:
        for name, value in actual.state_dict().items():
            torch.testing.assert_close(value, expected.state_dict()[name], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("body_learning_rate", [None, 0.0, 0.01])
@pytest.mark.parametrize("l2_weight", [None, 0.0, 0.2])
def test_optimizer_defaults_only_for_none(tiny_model, body_learning_rate, l2_weight):
    optimizer = tiny_model._prepare_optimizer(0.1, body_learning_rate, l2_weight)
    assert [group["lr"] for group in optimizer.param_groups] == [
        0.1 if body_learning_rate is None else body_learning_rate,
        0.1,
    ]
    assert all(group["weight_decay"] == (0.01 if l2_weight is None else l2_weight) for group in optimizer.param_groups)
