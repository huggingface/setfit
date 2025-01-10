from pathlib import Path

import datasets
import pytest
from datasets import Dataset, load_dataset
from packaging.version import Version, parse

from setfit import SetFitModel, SetFitModelCardData, Trainer, TrainingArguments
from setfit.data import sample_dataset
from setfit.model_card import generate_model_card, is_on_huggingface

from .model_card_pattern import MODEL_CARD_PATTERN


def test_model_card(tmp_path: Path) -> None:
    dataset = load_dataset("sst2")
    train_dataset = sample_dataset(dataset["train"], label_column="label", num_samples=8)
    eval_dataset = dataset["validation"].select(range(10))
    model = SetFitModel.from_pretrained(
        "sentence-transformers/paraphrase-albert-small-v2",
        labels=["negative", "positive"],
        model_card_data=SetFitModelCardData(
            model_id="tomaarsen/setfit-paraphrase-albert-small-v2-sst2",
            dataset_id="sst2",
            dataset_name="SST2",
            language=["en"],
            license="apache-2.0",
        ),
    )

    args = TrainingArguments(
        str(tmp_path),
        report_to="codecarbon",
        batch_size=1,
        eval_steps=1,
        logging_steps=1,
        max_steps=2,
        eval_strategy="steps",
        save_strategy="no",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        column_mapping={"sentence": "text"},
    )
    trainer.train()
    trainer.evaluate()
    model_card = generate_model_card(trainer.model)
    assert MODEL_CARD_PATTERN.fullmatch(model_card)


def test_model_card_languages() -> None:
    model = SetFitModel.from_pretrained(
        "sentence-transformers/paraphrase-albert-small-v2",
        model_card_data=SetFitModelCardData(
            language=["en", "nl", "de"],
        ),
    )
    model_card = model.generate_model_card()
    assert "**Languages:** en, nl, de" in model_card


def test_is_on_huggingface_edge_case() -> None:
    assert not is_on_huggingface("test_value")
    assert not is_on_huggingface("a/test/value")


@pytest.mark.skipif(
    parse(datasets.__version__) < Version("2.14.0"), reason="Inferring dataset_id only works from datasets >= 2.14.0"
)
@pytest.mark.parametrize("dataset_id", ("SetFit/emotion", "SetFit/sst2"))
def test_infer_dataset_id(dataset_id: str) -> None:
    model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-albert-small-v2")
    train_dataset = load_dataset(dataset_id, split="train")

    # This triggers inferring the dataset_id from train_dataset
    Trainer(model=model, train_dataset=train_dataset)
    assert model.model_card_data.dataset_id == dataset_id


def test_cant_infer_dataset_id():
    model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-albert-small-v2")
    train_dataset = Dataset.from_dict({"text": ["a", "b", "c", "d"], "label": [0, 1, 1, 0]})

    # This triggers inferring the dataset_id from train_dataset
    Trainer(model=model, train_dataset=train_dataset)
    assert model.model_card_data.dataset_id is None
