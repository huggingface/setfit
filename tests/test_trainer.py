import pytest
from datasets import Dataset

from setfit.modeling import SetFitModel
from setfit.trainer import SetFitTrainer


@pytest.fixture
def setup_trainer():
    model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-albert-small-v2")
    trainer = SetFitTrainer(
        model=model,
        train_dataset=Dataset.from_dict({"text_new": ["a", "b", "c"], "label_new": [0, 1, 2]}),
    )
    return trainer


def test_column_mapping_is_valid(setup_trainer):
    trainer = setup_trainer
    trainer.column_mapping = {"text_new": "text", "label_new": "label"}
    trainer._validate_column_mapping(trainer.train_dataset)
    formatted_dataset = trainer._apply_column_mapping(trainer.train_dataset, trainer.column_mapping)
    assert formatted_dataset.column_names == ["text", "label"]


def test_column_mapping_is_none(setup_trainer):
    trainer = setup_trainer
    trainer.column_mapping = None
    with pytest.raises(ValueError):
        trainer._validate_column_mapping(trainer.train_dataset)


def test_column_mapping_with_missing_label(setup_trainer):
    trainer = setup_trainer
    trainer.column_mapping = {"text_new": "text"}
    with pytest.raises(ValueError):
        trainer._validate_column_mapping(trainer.train_dataset)


def test_column_mapping_with_missing_text(setup_trainer):
    trainer = setup_trainer
    trainer.column_mapping = {"label_new": "label"}
    with pytest.raises(ValueError):
        trainer._validate_column_mapping(trainer.train_dataset)
