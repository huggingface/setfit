from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch

from setfit import AbsaModel
from setfit.span.aspect_extractor import AspectExtractor
from setfit.span.modeling import AspectModel, PolarityModel


def test_loading():
    model = AbsaModel.from_pretrained("sentence-transformers/paraphrase-albert-small-v2")
    assert isinstance(model, AbsaModel)
    assert isinstance(model.aspect_extractor, AspectExtractor)
    assert isinstance(model.aspect_model, AspectModel)
    assert isinstance(model.polarity_model, PolarityModel)

    model = AbsaModel.from_pretrained(
        "sentence-transformers/paraphrase-albert-small-v2@6c91e73a51599e35bd1145dfdcd3289215225009",
        "sentence-transformers/paraphrase-albert-small-v2",
    )
    assert isinstance(model, AbsaModel)

    model = AbsaModel.from_pretrained(
        "sentence-transformers/paraphrase-albert-small-v2",
        "sentence-transformers/paraphrase-albert-small-v2@6c91e73a51599e35bd1145dfdcd3289215225009",
    )
    assert isinstance(model, AbsaModel)

    with pytest.raises(OSError):
        model = AbsaModel.from_pretrained(
            "sentence-transformers/paraphrase-albert-small-v2", spacy_model="not_a_spacy_model"
        )

    model = AbsaModel.from_pretrained("sentence-transformers/paraphrase-albert-small-v2", normalize_embeddings=True)
    assert model.aspect_model.normalize_embeddings
    assert model.polarity_model.normalize_embeddings

    aspect_model = AspectModel.from_pretrained("sentence-transformers/paraphrase-albert-small-v2", span_context=12)
    assert aspect_model.span_context == 12
    polarity_model = PolarityModel.from_pretrained("sentence-transformers/paraphrase-albert-small-v2", span_context=12)
    assert polarity_model.span_context == 12

    model = AbsaModel.from_pretrained("sentence-transformers/paraphrase-albert-small-v2", span_contexts=(12, None))
    assert model.aspect_model.span_context == 12
    assert model.polarity_model.span_context == 3  # <- default


def test_save_load(absa_model: AbsaModel) -> None:
    absa_model.polarity_model.span_context = 5

    with TemporaryDirectory() as tmp_dir:
        tmp_dir = str(Path(tmp_dir) / "model")
        absa_model.save_pretrained(tmp_dir)
        assert (Path(tmp_dir + "-aspect") / "config_span_setfit.json").exists()
        assert (Path(tmp_dir + "-polarity") / "config_span_setfit.json").exists()

        fresh_model = AbsaModel.from_pretrained(tmp_dir + "-aspect", tmp_dir + "-polarity")
        assert fresh_model.polarity_model.span_context == 5

    with TemporaryDirectory() as aspect_tmp_dir:
        with TemporaryDirectory() as polarity_tmp_dir:
            absa_model.save_pretrained(aspect_tmp_dir, polarity_tmp_dir)
            assert (Path(aspect_tmp_dir) / "config_span_setfit.json").exists()
            assert (Path(polarity_tmp_dir) / "config_span_setfit.json").exists()

            fresh_model = AbsaModel.from_pretrained(aspect_tmp_dir, polarity_tmp_dir)
            assert fresh_model.polarity_model.span_context == 5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA must be available to move a model between devices")
def test_to(absa_model: AbsaModel) -> None:
    assert absa_model.device.type == "cuda"
    absa_model.to("cpu")
    assert absa_model.device.type == "cpu"
    assert absa_model.aspect_model.device.type == "cpu"
    assert absa_model.polarity_model.device.type == "cpu"
