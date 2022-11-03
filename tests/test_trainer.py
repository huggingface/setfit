from unittest import TestCase

import evaluate
import pytest
from datasets import Dataset
from transformers.testing_utils import require_optuna
from transformers.utils.hp_naming import TrialShortNamer

from setfit.modeling import SetFitModel
from setfit.trainer import SetFitTrainer
from setfit.utils import BestRun


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

    def test_trainer_support_evaluate_kwargs_when_metric_is_str(self):
        dataset = Dataset.from_dict(
            {"text_new": ["a", "b", "c"], "label_new": [0, 1, 2], "extra_column": ["d", "e", "f"]}
        )

        trainer = SetFitTrainer(
            model=self.model,
            train_dataset=dataset,
            eval_dataset=dataset,
            metric="f1",
            num_iterations=self.num_iterations,
            column_mapping={"text_new": "text", "label_new": "label"},
        )

        trainer.train()
        metrics = trainer.evaluate(average="micro")

        self.assertEqual(metrics["f1"], 1.0)

    def test_trainer_support_callable_as_metric(self):
        dataset = Dataset.from_dict(
            {"text_new": ["a", "b", "c"], "label_new": [0, 1, 2], "extra_column": ["d", "e", "f"]}
        )

        f1_metric = evaluate.load("f1")
        accuracy_metric = evaluate.load("accuracy")

        def compute_metrics(y_pred, y_test):
            return {
                "f1": f1_metric.compute(predictions=y_pred, references=y_test, average="micro")["f1"],
                "accuracy": accuracy_metric.compute(predictions=y_pred, references=y_test)["accuracy"],
            }

        trainer = SetFitTrainer(
            model=self.model,
            train_dataset=dataset,
            eval_dataset=dataset,
            metric=compute_metrics,
            num_iterations=self.num_iterations,
            column_mapping={"text_new": "text", "label_new": "label"},
        )

        trainer.train()
        metrics = trainer.evaluate()

        self.assertEqual(
            {
                "f1": 1.0,
                "accuracy": 1.0,
            },
            metrics,
        )

    def test_raise_when_metric_value_is_invalid(self):
        dataset = Dataset.from_dict(
            {"text_new": ["a", "b", "c"], "label_new": [0, 1, 2], "extra_column": ["d", "e", "f"]}
        )

        trainer = SetFitTrainer(
            model=self.model,
            train_dataset=dataset,
            eval_dataset=dataset,
            metric=42,  # invalid metric value
            num_iterations=self.num_iterations,
            column_mapping={"text_new": "text", "label_new": "label"},
        )

        trainer.train()

        with self.assertRaises(ValueError):
            trainer.evaluate()

    def test_trainer_raises_error_with_wrong_warmup_proportion(self):
        # warmup_proportion must not be > 1.0
        with pytest.raises(ValueError):
            SetFitTrainer(warmup_proportion=1.1)

        # warmup_proportion must not be < 0.0
        with pytest.raises(ValueError):
            SetFitTrainer(warmup_proportion=-0.1)


class SetFitTrainerMultilabelTest(TestCase):
    def setUp(self):
        self.model = SetFitModel.from_pretrained(
            "sentence-transformers/paraphrase-albert-small-v2", multi_target_strategy="one-vs-rest"
        )
        self.num_iterations = 1

    def test_trainer_multilabel_support_evaluate_kwargs_when_metric_is_str(self):
        dataset = Dataset.from_dict({"text_new": ["a", "b", "c"], "label_new": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]})

        trainer = SetFitTrainer(
            model=self.model,
            train_dataset=dataset,
            eval_dataset=dataset,
            metric="f1",
            num_iterations=self.num_iterations,
            column_mapping={"text_new": "text", "label_new": "label"},
        )

        trainer.train()
        metrics = trainer.evaluate(average="micro")

        self.assertEqual(metrics["f1"], 1.0)

    def test_trainer_multilabel_support_callable_as_metric(self):
        dataset = Dataset.from_dict({"text_new": ["a", "b", "c"], "label_new": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]})

        multilabel_f1_metric = evaluate.load("f1", "multilabel")
        multilabel_accuracy_metric = evaluate.load("accuracy", "multilabel")

        def compute_metrics(y_pred, y_test):
            return {
                "f1": multilabel_f1_metric.compute(predictions=y_pred, references=y_test, average="micro")["f1"],
                "accuracy": multilabel_accuracy_metric.compute(predictions=y_pred, references=y_test)["accuracy"],
            }

        trainer = SetFitTrainer(
            model=self.model,
            train_dataset=dataset,
            eval_dataset=dataset,
            metric=compute_metrics,
            num_iterations=self.num_iterations,
            column_mapping={"text_new": "text", "label_new": "label"},
        )

        trainer.train()
        metrics = trainer.evaluate()

        self.assertEqual(
            {
                "f1": 1.0,
                "accuracy": 1.0,
            },
            metrics,
        )


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
                "max_iter": trial.suggest_int("max_iter", 50, 300),
                "solver": trial.suggest_categorical("solver", ["newton-cg", "lbfgs", "liblinear"]),
            }

        def model_init(params):
            params = params or {}
            max_iter = params.get("max_iter", 100)
            solver = params.get("solver", "liblinear")
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
        result = trainer.hyperparameter_search(direction="minimize", hp_space=hp_space, hp_name=hp_name, n_trials=4)
        assert isinstance(result, BestRun)
        assert result.hyperparameters.keys() == {"learning_rate", "batch_size", "max_iter", "solver"}
