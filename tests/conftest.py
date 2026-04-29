import pytest

from setfit import SetFitModel


@pytest.fixture()
def model() -> SetFitModel:
    return SetFitModel.from_pretrained("sentence-transformers/paraphrase-albert-small-v2")
