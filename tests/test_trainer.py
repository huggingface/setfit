from unittest import TestCase

import pytest
from datasets import Dataset

from setfit.modeling import SetFitModel
from setfit.trainer import SetFitTrainer


class SetFitTrainerTest(TestCase):
    def setUp(self):
        self.model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-albert-small-v2")
        self.num_iterations = 1

    def test_trainer_works_with_model_init(self):
        def get_model():
            model_name = "sentence-transformers/paraphrase-albert-small-v2"
            return SetFitModel.from_pretrained(model_name)

        dataset = Dataset.from_dict(
            {"text_new": ["a", "b", "c"], "label_new": [0, 1, 2], "extra_column": ["d", "e", "f"]}
        )
        trainer = SetFitTrainer(
            model_init=get_model,
            train_dataset=dataset,
            eval_dataset=dataset,
            num_iterations=self.num_iterations,
            column_mapping={"text_new": "text", "label_new": "label"},
        )
        trainer.train()
        metrics = trainer.evaluate()
        self.assertEqual(metrics["accuracy"], 1.0)

    def test_trainer_works_with_column_mapping(self):
        dataset = Dataset.from_dict(
            {"text_new": ["a", "b", "c"], "label_new": [0, 1, 2], "extra_column": ["d", "e", "f"]}
        )
        trainer = SetFitTrainer(
            model=self.model,
            train_dataset=dataset,
            eval_dataset=dataset,
            num_iterations=self.num_iterations,
            column_mapping={"text_new": "text", "label_new": "label"},
        )
        trainer.train()
        metrics = trainer.evaluate()
        self.assertEqual(metrics["accuracy"], 1.0)

    def test_trainer_works_with_default_columns(self):
        dataset = Dataset.from_dict({"text": ["a", "b", "c"], "label": [0, 1, 2], "extra_column": ["d", "e", "f"]})
        trainer = SetFitTrainer(
            model=self.model, train_dataset=dataset, eval_dataset=dataset, num_iterations=self.num_iterations
        )
        trainer.train()
        metrics = trainer.evaluate()
        self.assertEqual(metrics["accuracy"], 1.0)

    def test_trainer_raises_error_with_missing_label(self):
        dataset = Dataset.from_dict({"text": ["a", "b", "c"], "extra_column": ["d", "e", "f"]})
        trainer = SetFitTrainer(
            model=self.model, train_dataset=dataset, eval_dataset=dataset, num_iterations=self.num_iterations
        )
        with pytest.raises(ValueError):
            trainer.train()

    def test_trainer_raises_error_with_missing_text(self):
        dataset = Dataset.from_dict({"label": [0, 1, 2], "extra_column": ["d", "e", "f"]})
        trainer = SetFitTrainer(
            model=self.model, train_dataset=dataset, eval_dataset=dataset, num_iterations=self.num_iterations
        )
        with pytest.raises(ValueError):
            trainer.train()
