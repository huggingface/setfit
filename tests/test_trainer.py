from unittest import TestCase

import pytest
from datasets import Dataset
from transformers.testing_utils import require_optuna
from transformers.utils.hp_naming import TrialShortNamer

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

    def test_column_mapping_with_missing_text(self):
        dataset = Dataset.from_dict({"text": ["a", "b", "c"], "extra_column": ["d", "e", "f"]})
        trainer = SetFitTrainer(
            model=self.model,
            train_dataset=dataset,
            eval_dataset=dataset,
            num_iterations=self.num_iterations,
            column_mapping={"label_new": "label"},
        )
        with pytest.raises(ValueError):
            trainer._validate_column_mapping(trainer.train_dataset)

    def test_column_mapping_multilabel(self):
        dataset = Dataset.from_dict({"text_new": ["a", "b", "c"], "label_new": [[0, 1], [1, 2], [2, 0]]})

        trainer = SetFitTrainer(
            model=self.model,
            train_dataset=dataset,
            eval_dataset=dataset,
            num_iterations=self.num_iterations,
            column_mapping={"text_new": "text", "label_new": "label"},
        )

        trainer._validate_column_mapping(trainer.train_dataset)
        formatted_dataset = trainer._apply_column_mapping(trainer.train_dataset, trainer.column_mapping)

        assert formatted_dataset.column_names == ["text", "label"]

        assert formatted_dataset[0]["text"] == "a"
        assert formatted_dataset[0]["label"] == [0, 1]

        assert formatted_dataset[1]["text"] == "b"


@require_optuna
class TrainerHyperParameterOptunaIntegrationTest(TestCase):
    def setUp(self):
        self.dataset = Dataset.from_dict(
            {"text_new": ["a", "b", "c"], "label_new": [0, 1, 2], "extra_column": ["d", "e", "f"]}
        )
        self.num_iterations = 1

    def test_hyperparameter_search(self):
        class MyTrialShortNamer(TrialShortNamer):
            DEFAULTS = {"max_iter": 100, "solver": "liblinear"}

        def hp_space(trial):
            return {
                "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16, 32, 64]),
            }

        def model_init(trial):
            if trial is not None:
                max_iter = trial.suggest_int("max_iter", 50, 300)
                solver = trial.suggest_categorical("solver", ["newton-cg", "lbfgs", "liblinear"])
            else:
                max_iter = 100
                solver = "liblinear"
            params = {
                "head_params": {
                    "max_iter": max_iter,
                    "solver": solver,
                }
            }
            return SetFitModel.from_pretrained("sentence-transformers/paraphrase-albert-small-v2", **params)

        def hp_name(trial):
            return MyTrialShortNamer.shortname(trial.params)

        trainer = SetFitTrainer(
            train_dataset=self.dataset,
            eval_dataset=self.dataset,
            num_iterations=self.num_iterations,
            model_init=model_init,
            column_mapping={"text_new": "text", "label_new": "label"},
        )
        trainer.hyperparameter_search(direction="minimize", hp_space=hp_space, hp_name=hp_name, n_trials=4)
