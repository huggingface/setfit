import pytest
from datasets import Dataset

from setfit import AbsaModel, SetFitModel


@pytest.fixture()
def model() -> SetFitModel:
    return SetFitModel.from_pretrained("sentence-transformers/paraphrase-albert-small-v2")


@pytest.fixture()
def absa_model() -> AbsaModel:
    return AbsaModel.from_pretrained("sentence-transformers/paraphrase-albert-small-v2")


@pytest.fixture()
def absa_dataset() -> Dataset:
    texts = [
        "It is about food and ambiance, and imagine how dreadful it will be it we only had to listen to an idle engine.",
        "It is about food and ambiance, and imagine how dreadful it will be it we only had to listen to an idle engine.",
        "Food is great and inexpensive.",
        "Good bagels and good cream cheese.",
        "Good bagels and good cream cheese.",
    ]
    spans = ["food", "ambiance", "Food", "bagels", "cream cheese"]
    labels = ["negative", "negative", "positive", "positive", "positive"]
    ordinals = [0, 0, 0, 0, 0]
    return Dataset.from_dict({"text": texts, "span": spans, "label": labels, "ordinal": ordinals})
