import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch
from datasets import Dataset
from pytest import LogCaptureFixture

from setfit import AbsaModel
from setfit.logging import get_logger
from setfit.span.aspect_extractor import AspectExtractor
from setfit.span.modeling import AspectModel, PolarityModel
from tests.test_modeling import torch_cuda_available


def test_loading():
    model = AbsaModel.from_pretrained("sentence-transformers/paraphrase-albert-small-v2", spacy_model="en_core_web_sm")
    assert isinstance(model, AbsaModel)
    assert isinstance(model.aspect_extractor, AspectExtractor)
    assert isinstance(model.aspect_model, AspectModel)
    assert isinstance(model.polarity_model, PolarityModel)

    model = AbsaModel.from_pretrained(
        "sentence-transformers/paraphrase-albert-small-v2@6c91e73a51599e35bd1145dfdcd3289215225009",
        "sentence-transformers/paraphrase-albert-small-v2",
        spacy_model="en_core_web_sm",
    )
    assert isinstance(model, AbsaModel)

    model = AbsaModel.from_pretrained(
        "sentence-transformers/paraphrase-albert-small-v2",
        "sentence-transformers/paraphrase-albert-small-v2@6c91e73a51599e35bd1145dfdcd3289215225009",
        spacy_model="en_core_web_sm",
    )
    assert isinstance(model, AbsaModel)

    with pytest.raises(OSError):
        model = AbsaModel.from_pretrained(
            "sentence-transformers/paraphrase-albert-small-v2", spacy_model="not_a_spacy_model"
        )

    model = AbsaModel.from_pretrained(
        "sentence-transformers/paraphrase-albert-small-v2", spacy_model="en_core_web_sm", normalize_embeddings=True
    )
    assert model.aspect_model.normalize_embeddings
    assert model.polarity_model.normalize_embeddings

    aspect_model = AspectModel.from_pretrained("sentence-transformers/paraphrase-albert-small-v2", span_context=12)
    assert aspect_model.span_context == 12
    polarity_model = PolarityModel.from_pretrained("sentence-transformers/paraphrase-albert-small-v2", span_context=12)
    assert polarity_model.span_context == 12

    model = AbsaModel.from_pretrained(
        "sentence-transformers/paraphrase-albert-small-v2", spacy_model="en_core_web_sm", span_contexts=(12, 4)
    )
    assert model.aspect_model.span_context == 12
    assert model.polarity_model.span_context == 4


def test_save_load(absa_model: AbsaModel, caplog: LogCaptureFixture) -> None:
    logger = get_logger("setfit")
    logger.propagate = True

    absa_model.polarity_model.span_context = 5

    with TemporaryDirectory() as tmp_dir:
        tmp_dir = str(Path(tmp_dir) / "model")
        absa_model.save_pretrained(tmp_dir)
        assert (Path(tmp_dir + "-aspect") / "config_setfit.json").exists()
        assert (Path(tmp_dir + "-polarity") / "config_setfit.json").exists()

        fresh_model = AbsaModel.from_pretrained(
            tmp_dir + "-aspect", tmp_dir + "-polarity", spacy_model="en_core_web_sm"
        )
        assert fresh_model.polarity_model.span_context == 5

        # We expect a warning if we override the configured data:
        AbsaModel.from_pretrained(tmp_dir + "-aspect", tmp_dir + "-polarity", span_contexts=[4, 4])
        log_texts = [record[2] for record in caplog.record_tuples]
        assert "Overriding span_context in model configuration from 0 to 4." in log_texts
        assert "Overriding span_context in model configuration from 5 to 4." in log_texts
        assert len(caplog.record_tuples) == 2
        caplog.clear()

        # Error because en_core_web_bla doesn't exist
        with pytest.raises(OSError):
            AbsaModel.from_pretrained(tmp_dir + "-aspect", tmp_dir + "-polarity", spacy_model="en_core_web_bla")
        log_texts = [record[2] for record in caplog.record_tuples]
        assert "Overriding spacy_model in model configuration from en_core_web_sm to en_core_web_bla." in log_texts
        assert "Overriding spacy_model in model configuration from en_core_web_sm to en_core_web_bla." in log_texts
        assert len(caplog.record_tuples) == 2
        caplog.clear()

    with TemporaryDirectory() as aspect_tmp_dir:
        with TemporaryDirectory() as polarity_tmp_dir:
            absa_model.save_pretrained(aspect_tmp_dir, polarity_tmp_dir)
            assert (Path(aspect_tmp_dir) / "config_setfit.json").exists()
            assert (Path(polarity_tmp_dir) / "config_setfit.json").exists()

            fresh_model = AbsaModel.from_pretrained(aspect_tmp_dir, polarity_tmp_dir)
            assert fresh_model.polarity_model.span_context == 5
            assert fresh_model.aspect_model.spacy_model == "en_core_web_sm"
            assert fresh_model.polarity_model.spacy_model == "en_core_web_sm"

            # Loading a model with different spacy_model settings
            polarity_config_path = str(Path(polarity_tmp_dir) / "config_setfit.json")
            with open(polarity_config_path, "r") as f:
                config = json.load(f)
            assert config == {
                "span_context": 5,
                "normalize_embeddings": False,
                "spacy_model": "en_core_web_sm",
                "labels": None,
            }
            config["spacy_model"] = "en_core_web_bla"
            with open(polarity_config_path, "w") as f:
                json.dump(config, f)
            # Load a model with the updated config, there should be a warning
            fresh_model = AbsaModel.from_pretrained(aspect_tmp_dir, polarity_tmp_dir)
            assert len(caplog.record_tuples) == 1
            assert caplog.record_tuples[0][2] == (
                "The Aspect and Polarity models are configured to use different spaCy models:\n"
                "* 'en_core_web_sm' for the aspect model, and\n"
                "* 'en_core_web_bla' for the polarity model.\n"
                "This model will use 'en_core_web_sm'."
            )

    logger.propagate = False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA must be available to move a model between devices")
def test_to(absa_model: AbsaModel) -> None:
    assert absa_model.device.type == "cuda"
    absa_model.to("cpu")
    assert absa_model.device.type == "cpu"
    assert absa_model.aspect_model.device.type == "cpu"
    assert absa_model.polarity_model.device.type == "cpu"


@torch_cuda_available
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_load_model_on_device(device):
    model = AbsaModel.from_pretrained("sentence-transformers/paraphrase-albert-small-v2", device=device)
    assert model.device.type == device
    assert model.polarity_model.device.type == device
    assert model.aspect_model.device.type == device


def test_predict_dataset(trained_absa_model: AbsaModel):
    inputs = Dataset.from_dict(
        {
            "text": [
                "But the staff was so horrible to us.",
                "To be completely fair, the only redeeming factor was the food, which was above average, but couldn't make up for all the other deficiencies of Teodora.",
                "The food is uniformly exceptional, with a very capable kitchen which will proudly whip up whatever you feel like eating, whether it's on the menu or not.",
                "The food is uniformly exceptional, with a very capable kitchen which will proudly whip up whatever you feel like eating, whether it's on the menu or not.",
                "The food is uniformly exceptional, with a very capable kitchen which will proudly whip up whatever you feel like eating, whether it's on the menu or not.",
            ],
            "span": ["staff", "food", "food", "kitchen", "menu"],
            "label": ["negative", "positive", "positive", "positive", "neutral"],
            "ordinal": [0, 0, 0, 0, 0],
        }
    )
    outputs = trained_absa_model.predict(inputs)
    assert isinstance(outputs, Dataset)
    assert set(outputs.column_names) == {"pred_polarity", "text", "span", "label", "ordinal"}

    inputs = Dataset.from_dict(
        {
            "text": [
                "But the staff was so horrible to us.",
                "To be completely fair, the only redeeming factor was the food, which was above average, but couldn't make up for all the other deficiencies of Teodora.",
                "The food is uniformly exceptional, with a very capable kitchen which will proudly whip up whatever you feel like eating, whether it's on the menu or not.",
                "The food is uniformly exceptional, with a very capable kitchen which will proudly whip up whatever you feel like eating, whether it's on the menu or not.",
                "The food is uniformly exceptional, with a very capable kitchen which will proudly whip up whatever you feel like eating, whether it's on the menu or not.",
            ],
            "span": ["staff", "food", "food", "kitchen", "menu"],
        }
    )
    outputs = trained_absa_model.predict(inputs)
    assert isinstance(outputs, Dataset)
    assert "pred_polarity" in outputs.column_names


def test_predict_dataset_errors(trained_absa_model: AbsaModel):
    inputs = Dataset.from_dict(
        {
            "text": [
                "But the staff was so horrible to us.",
                "To be completely fair, the only redeeming factor was the food, which was above average, but couldn't make up for all the other deficiencies of Teodora.",
                "The food is uniformly exceptional, with a very capable kitchen which will proudly whip up whatever you feel like eating, whether it's on the menu or not.",
                "The food is uniformly exceptional, with a very capable kitchen which will proudly whip up whatever you feel like eating, whether it's on the menu or not.",
                "The food is uniformly exceptional, with a very capable kitchen which will proudly whip up whatever you feel like eating, whether it's on the menu or not.",
            ],
        }
    )
    with pytest.raises(
        ValueError,
        match=re.escape(
            "`inputs` must be either a `str`, a `List[str]`, or a `datasets.Dataset` with columns `text` and `span` and optionally `ordinal`. "
            "Found a dataset with these columns: ['text']."
        ),
    ):
        trained_absa_model.predict(inputs)

    inputs = Dataset.from_dict(
        {
            "text": [
                "But the staff was so horrible to us.",
                "To be completely fair, the only redeeming factor was the food, which was above average, but couldn't make up for all the other deficiencies of Teodora.",
                "The food is uniformly exceptional, with a very capable kitchen which will proudly whip up whatever you feel like eating, whether it's on the menu or not.",
                "The food is uniformly exceptional, with a very capable kitchen which will proudly whip up whatever you feel like eating, whether it's on the menu or not.",
                "The food is uniformly exceptional, with a very capable kitchen which will proudly whip up whatever you feel like eating, whether it's on the menu or not.",
            ],
            "span": ["staff", "food", "food", "kitchen", "menu"],
            "pred_polarity": ["negative", "positive", "positive", "positive", "neutral"],
        }
    )
    with pytest.raises(
        ValueError,
        match=re.escape(
            "`predict_dataset` wants to add a `pred_polarity` column, but the input dataset already contains that column."
        ),
    ):
        trained_absa_model.predict(inputs)
