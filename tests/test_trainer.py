import os
import re
import tempfile
from pathlib import Path
from unittest import TestCase

import evaluate
import pytest
import torch
from datasets import Dataset, load_dataset
from pytest import LogCaptureFixture
from sentence_transformers import losses
from transformers import TrainerCallback
from transformers import TrainingArguments as TransformersTrainingArguments
from transformers.testing_utils import require_optuna
from transformers.utils.hp_naming import TrialShortNamer

from setfit import logging
from setfit.losses import SupConLoss
from setfit.modeling import SetFitModel
from setfit.trainer import Trainer
from setfit.training_args import TrainingArguments
from setfit.utils import BestRun


logging.set_verbosity_warning()
logging.enable_propagation()


class TrainerTest(TestCase):
    def setUp(self):
        self.model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-albert-small-v2")
        self.args = TrainingArguments(num_iterations=1)

    def test_trainer_works_with_model_init(self):
        def get_model():
            model_name = "sentence-transformers/paraphrase-albert-small-v2"
            return SetFitModel.from_pretrained(model_name)

        dataset = Dataset.from_dict(
            {"text_new": ["a", "b", "c"], "label_new": [0, 1, 2], "extra_column": ["d", "e", "f"]}
        )
        trainer = Trainer(
            model_init=get_model,
            args=self.args,
            train_dataset=dataset,
            eval_dataset=dataset,
            column_mapping={"text_new": "text", "label_new": "label"},
        )
        trainer.train()
        metrics = trainer.evaluate()
        self.assertEqual(metrics["accuracy"], 1.0)

    def test_trainer_works_with_column_mapping(self):
        dataset = Dataset.from_dict(
            {"text_new": ["a", "b", "c"], "label_new": [0, 1, 2], "extra_column": ["d", "e", "f"]}
        )
        trainer = Trainer(
            model=self.model,
            args=self.args,
            train_dataset=dataset,
            eval_dataset=dataset,
            column_mapping={"text_new": "text", "label_new": "label"},
        )
        trainer.train()
        metrics = trainer.evaluate()
        self.assertEqual(metrics["accuracy"], 1.0)

    def test_trainer_works_with_default_columns(self):
        dataset = Dataset.from_dict({"text": ["a", "b", "c"], "label": [0, 1, 2], "extra_column": ["d", "e", "f"]})
        trainer = Trainer(model=self.model, args=self.args, train_dataset=dataset, eval_dataset=dataset)
        trainer.train()
        metrics = trainer.evaluate()
        self.assertEqual(metrics["accuracy"], 1.0)

    def test_trainer_works_with_alternate_dataset_for_evaluate(self):
        dataset = Dataset.from_dict({"text": ["a", "b", "c"], "label": [0, 1, 2], "extra_column": ["d", "e", "f"]})
        alternate_dataset = Dataset.from_dict(
            {"text": ["x", "y", "z"], "label": [0, 1, 2], "extra_column": ["d", "e", "f"]}
        )
        trainer = Trainer(model=self.model, args=self.args, train_dataset=dataset, eval_dataset=dataset)
        trainer.train()
        metrics = trainer.evaluate(alternate_dataset)
        self.assertNotEqual(metrics["accuracy"], 1.0)

    def test_trainer_raises_error_with_missing_label(self):
        dataset = Dataset.from_dict({"text": ["a", "b", "c"], "extra_column": ["d", "e", "f"]})
        trainer = Trainer(model=self.model, args=self.args, train_dataset=dataset, eval_dataset=dataset)
        with pytest.raises(ValueError):
            trainer.train()

    def test_trainer_raises_error_with_missing_text(self):
        """If the required columns are missing from the dataset, the library should throw an error and list the columns found."""
        dataset = Dataset.from_dict({"label": [0, 1, 2], "extra_column": ["d", "e", "f"]})
        trainer = Trainer(model=self.model, args=self.args, train_dataset=dataset, eval_dataset=dataset)
        expected_message = re.escape(
            "SetFit expected the dataset to have the columns ['label', 'text'], "
            "but only the columns ['extra_column', 'label'] were found. "
            "Either make sure these columns are present, or specify which columns to use with column_mapping in Trainer."
        )
        with pytest.raises(ValueError, match=expected_message):
            trainer._validate_column_mapping(trainer.train_dataset)

    def test_column_mapping_raises_error_when_mapped_columns_missing(self):
        """If the columns specified in the column mapping are missing from the dataset, the library should throw an error and list the columns found."""
        dataset = Dataset.from_dict({"text": ["a", "b", "c"], "extra_column": ["d", "e", "f"]})
        trainer = Trainer(
            model=self.model,
            args=self.args,
            train_dataset=dataset,
            eval_dataset=dataset,
            column_mapping={"text_new": "text", "label_new": "label"},
        )
        expected_message = re.escape(
            "The column mapping expected the columns ['label_new', 'text_new'] in the dataset, "
            "but the dataset had the columns ['extra_column', 'text'].",
        )
        with pytest.raises(ValueError, match=expected_message):
            trainer._validate_column_mapping(trainer.train_dataset)

    def test_trainer_raises_error_when_dataset_not_split(self):
        """Verify that an error is raised if we pass an unsplit dataset to the trainer."""
        dataset = Dataset.from_dict({"text": ["a", "b", "c", "d"], "label": [0, 0, 1, 1]}).train_test_split(
            test_size=0.5
        )
        trainer = Trainer(model=self.model, args=self.args, train_dataset=dataset, eval_dataset=dataset)
        expected_message = re.escape(
            "SetFit expected a Dataset, but it got a DatasetDict with the splits ['test', 'train']. "
            "Did you mean to select one of these splits from the dataset?",
        )
        with pytest.raises(ValueError, match=expected_message):
            trainer._validate_column_mapping(trainer.train_dataset)

    def test_trainer_raises_error_when_dataset_is_dataset_dict_with_train(self):
        """Verify that a useful error is raised if we pass an unsplit dataset with only a `train` split to the trainer."""
        with tempfile.TemporaryDirectory() as tmpdirname:
            path = Path(tmpdirname) / "test_dataset_dict_with_train.csv"
            path.write_text("label,text\n1,good\n0,terrible\n")
            dataset = load_dataset("csv", data_files=str(path))
        trainer = Trainer(model=self.model, args=self.args, train_dataset=dataset, eval_dataset=dataset)
        expected_message = re.escape(
            "SetFit expected a Dataset, but it got a DatasetDict with the split ['train']. "
            "Did you mean to select the training split with dataset['train']?",
        )
        with pytest.raises(ValueError, match=expected_message):
            trainer._validate_column_mapping(trainer.train_dataset)

    def test_column_mapping_multilabel(self):
        dataset = Dataset.from_dict({"text_new": ["a", "b", "c"], "label_new": [[0, 1], [1, 2], [2, 0]]})

        trainer = Trainer(
            model=self.model,
            args=self.args,
            train_dataset=dataset,
            eval_dataset=dataset,
            column_mapping={"text_new": "text", "label_new": "label"},
        )

        trainer._validate_column_mapping(trainer.train_dataset)
        formatted_dataset = trainer._apply_column_mapping(trainer.train_dataset, trainer.column_mapping)

        assert formatted_dataset.column_names == ["text", "label"]

        assert formatted_dataset[0]["text"] == "a"
        assert formatted_dataset[0]["label"] == [0, 1]

        assert formatted_dataset[1]["text"] == "b"

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

        trainer = Trainer(
            model=self.model,
            args=self.args,
            train_dataset=dataset,
            eval_dataset=dataset,
            metric=compute_metrics,
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

        trainer = Trainer(
            model=self.model,
            args=self.args,
            train_dataset=dataset,
            eval_dataset=dataset,
            metric="this-metric-does-not-exist",  # invalid metric value
            column_mapping={"text_new": "text", "label_new": "label"},
        )

        trainer.train()

        with self.assertRaises(FileNotFoundError):
            trainer.evaluate()


class TrainerDifferentiableHeadTest(TestCase):
    def setUp(self):
        self.dataset = Dataset.from_dict(
            {"text_new": ["a", "b", "c"], "label_new": [0, 1, 2], "extra_column": ["d", "e", "f"]}
        )
        self.model = SetFitModel.from_pretrained(
            "sentence-transformers/paraphrase-albert-small-v2",
            use_differentiable_head=True,
            head_params={"out_features": 3},
        )
        self.args = TrainingArguments(num_iterations=1)

    def test_trainer_max_length_exceeds_max_acceptable_length(self):
        trainer = Trainer(
            model=self.model,
            args=self.args,
            train_dataset=self.dataset,
            eval_dataset=self.dataset,
            column_mapping={"text_new": "text", "label_new": "label"},
        )
        trainer.unfreeze(keep_body_frozen=True)
        with self.assertLogs(level=logging.WARNING) as cm:
            max_length = 4096
            max_acceptable_length = self.model.model_body.get_max_seq_length()
            args = TrainingArguments(num_iterations=1, max_length=max_length)
            trainer.train(args)
            self.assertEqual(
                cm.output,
                [
                    (
                        f"WARNING:setfit.modeling:The specified `max_length`: {max_length} is greater than the maximum length "
                        f"of the current model body: {max_acceptable_length}. Using {max_acceptable_length} instead."
                    )
                ],
            )

    def test_trainer_max_length_is_smaller_than_max_acceptable_length(self):
        trainer = Trainer(
            model=self.model,
            args=self.args,
            train_dataset=self.dataset,
            eval_dataset=self.dataset,
            column_mapping={"text_new": "text", "label_new": "label"},
        )

        # An alternative way of `assertNoLogs`, which is new in Python 3.10
        try:
            with self.assertLogs(level=logging.WARNING) as cm:
                max_length = 32
                args = TrainingArguments(num_iterations=1, max_length=max_length)
                trainer.train(args)
                self.assertEqual(cm.output, [])
        except AssertionError as e:
            if e.args[0] != "no logs of level WARNING or higher triggered on root":
                raise AssertionError(e)


class TrainerMultilabelTest(TestCase):
    def setUp(self):
        self.model = SetFitModel.from_pretrained(
            "sentence-transformers/paraphrase-albert-small-v2", multi_target_strategy="one-vs-rest"
        )
        self.args = TrainingArguments(num_iterations=1)

    def test_trainer_multilabel_support_callable_as_metric(self):
        dataset = Dataset.from_dict({"text_new": ["a", "b", "c"], "label_new": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]})

        multilabel_f1_metric = evaluate.load("f1", "multilabel")
        multilabel_accuracy_metric = evaluate.load("accuracy", "multilabel")

        def compute_metrics(y_pred, y_test):
            return {
                "f1": multilabel_f1_metric.compute(predictions=y_pred, references=y_test, average="micro")["f1"],
                "accuracy": multilabel_accuracy_metric.compute(predictions=y_pred, references=y_test)["accuracy"],
            }

        trainer = Trainer(
            model=self.model,
            args=self.args,
            train_dataset=dataset,
            eval_dataset=dataset,
            metric=compute_metrics,
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


class TrainerMultilabelDifferentiableTest(TestCase):
    def setUp(self):
        self.model = SetFitModel.from_pretrained(
            "sentence-transformers/paraphrase-albert-small-v2",
            multi_target_strategy="one-vs-rest",
            use_differentiable_head=True,
            head_params={"out_features": 2},
        )
        self.args = TrainingArguments(num_iterations=1)

    def test_trainer_multilabel_support_callable_as_metric(self):
        dataset = Dataset.from_dict({"text_new": ["", "a", "b", "ab"], "label_new": [[0, 0], [1, 0], [0, 1], [1, 1]]})

        multilabel_f1_metric = evaluate.load("f1", "multilabel")
        multilabel_accuracy_metric = evaluate.load("accuracy", "multilabel")

        def compute_metrics(y_pred, y_test):
            return {
                "f1": multilabel_f1_metric.compute(predictions=y_pred, references=y_test, average="micro")["f1"],
                "accuracy": multilabel_accuracy_metric.compute(predictions=y_pred, references=y_test)["accuracy"],
            }

        trainer = Trainer(
            model=self.model,
            args=self.args,
            train_dataset=dataset,
            eval_dataset=dataset,
            metric=compute_metrics,
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
        self.args = TrainingArguments(num_iterations=1)

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

        trainer = Trainer(
            args=self.args,
            train_dataset=self.dataset,
            eval_dataset=self.dataset,
            model_init=model_init,
            column_mapping={"text_new": "text", "label_new": "label"},
        )
        result = trainer.hyperparameter_search(direction="minimize", hp_space=hp_space, hp_name=hp_name, n_trials=4)
        assert isinstance(result, BestRun)
        assert result.hyperparameters.keys() == {"learning_rate", "batch_size", "max_iter", "solver"}


# regression test for https://github.com/huggingface/setfit/issues/153
@pytest.mark.parametrize(
    "loss_class",
    [
        losses.BatchAllTripletLoss,
        losses.BatchHardTripletLoss,
        losses.BatchSemiHardTripletLoss,
        losses.BatchHardSoftMarginTripletLoss,
        SupConLoss,
    ],
)
def test_trainer_works_with_non_default_loss_class(loss_class):
    dataset = Dataset.from_dict({"text": ["a 1", "b 1", "c 1", "a 2", "b 2", "c 2"], "label": [0, 1, 2, 0, 1, 2]})
    model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-albert-small-v2")
    args = TrainingArguments(num_iterations=1, loss=loss_class)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        eval_dataset=dataset,
    )
    trainer.train()
    # no asserts here because this is a regression test - we only test if an exception is raised


def test_trainer_evaluate_with_strings(model: SetFitModel):
    dataset = Dataset.from_dict(
        {"text": ["positive sentence", "negative sentence"], "label": ["positive", "negative"]}
    )
    trainer = Trainer(
        model=model,
        args=TrainingArguments(num_iterations=1),
        train_dataset=dataset,
        eval_dataset=dataset,
    )
    trainer.train()
    # This used to fail due to "TypeError: can't convert np.ndarray of type numpy.str_.
    # The only supported types are: float64, float32, float16, complex64, complex128, int64, int32, int16, int8, uint8, and bool."
    model.predict(["another positive sentence"])


def test_trainer_evaluate_multilabel_f1():
    dataset = Dataset.from_dict({"text_new": ["", "a", "b", "ab"], "label_new": [[0, 0], [1, 0], [0, 1], [1, 1]]})
    model = SetFitModel.from_pretrained(
        "sentence-transformers/paraphrase-albert-small-v2", multi_target_strategy="one-vs-rest"
    )

    trainer = Trainer(
        model=model,
        args=TrainingArguments(num_iterations=5),
        train_dataset=dataset,
        eval_dataset=dataset,
        metric="f1",
        metric_kwargs={"average": "micro"},
        column_mapping={"text_new": "text", "label_new": "label"},
    )

    trainer.train()
    metrics = trainer.evaluate()
    assert metrics == {"f1": 1.0}


def test_trainer_evaluate_on_cpu() -> None:
    # This test used to fail if CUDA was available
    dataset = Dataset.from_dict({"text": ["positive sentence", "negative sentence"], "label": [1, 0]})
    model = SetFitModel.from_pretrained(
        "sentence-transformers/paraphrase-albert-small-v2", use_differentiable_head=True
    )

    def compute_metric(y_pred, y_test) -> None:
        assert y_pred.device == torch.device("cpu")
        return 1.0

    args = TrainingArguments(num_iterations=5)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        eval_dataset=dataset,
        metric=compute_metric,
    )
    trainer.train()
    trainer.evaluate()


def test_no_model_no_model_init():
    with pytest.raises(RuntimeError, match="`Trainer` requires either a `model` or `model_init` argument."):
        Trainer()


def test_model_and_model_init(model: SetFitModel):
    def model_init() -> SetFitModel:
        return model

    with pytest.raises(RuntimeError, match="`Trainer` requires either a `model` or `model_init` argument."):
        Trainer(model=model, model_init=model_init)


def test_trainer_callbacks(model: SetFitModel):
    trainer = Trainer(model=model)
    assert len(trainer.callback_handler.callbacks) >= 2
    callback_names = {callback.__class__.__name__ for callback in trainer.callback_handler.callbacks}
    assert {"DefaultFlowCallback", "ProgressCallback"} <= callback_names

    class TestCallback(TrainerCallback):
        pass

    callback = TestCallback()
    trainer.add_callback(callback)
    assert len(trainer.callback_handler.callbacks) == len(callback_names) + 1
    assert trainer.callback_handler.callbacks[-1] == callback

    assert trainer.pop_callback(callback) == callback
    trainer.add_callback(callback)
    assert trainer.callback_handler.callbacks[-1] == callback
    trainer.remove_callback(callback)
    assert callback not in trainer.callback_handler.callbacks


def test_trainer_warn_freeze(model: SetFitModel):
    trainer = Trainer(model)
    with pytest.warns(
        DeprecationWarning,
        match="Trainer.freeze` is deprecated and will be removed in v2.0.0 of SetFit. "
        "Please use `SetFitModel.freeze` directly instead.",
    ):
        trainer.freeze()


def test_train_with_kwargs(model: SetFitModel) -> None:
    train_dataset = Dataset.from_dict({"text": ["positive sentence", "negative sentence"], "label": [1, 0]})
    trainer = Trainer(model, train_dataset=train_dataset)
    with pytest.warns(DeprecationWarning, match="`Trainer.train` does not accept keyword arguments anymore."):
        trainer.train(num_epochs=5)


def test_train_no_dataset(model: SetFitModel) -> None:
    trainer = Trainer(model)
    with pytest.raises(ValueError, match="Training requires a `train_dataset` given to the `Trainer` initialization."):
        trainer.train()


def test_train_amp_save(model: SetFitModel, tmp_path: Path) -> None:
    args = TrainingArguments(output_dir=tmp_path, use_amp=True, save_steps=5, num_epochs=5)
    dataset = Dataset.from_dict({"text": ["a", "b", "c"], "label": [0, 1, 2]})
    trainer = Trainer(model, args=args, train_dataset=dataset, eval_dataset=dataset)
    trainer.train()
    assert trainer.evaluate() == {"accuracy": 1.0}
    assert os.listdir(tmp_path) == ["step_5"]


def test_train_load_best(model: SetFitModel, tmp_path: Path, caplog: LogCaptureFixture) -> None:
    args = TrainingArguments(
        output_dir=tmp_path,
        save_steps=5,
        eval_steps=5,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        num_epochs=5,
    )
    dataset = Dataset.from_dict({"text": ["a", "b", "c"], "label": [0, 1, 2]})
    trainer = Trainer(model, args=args, train_dataset=dataset, eval_dataset=dataset)
    with caplog.at_level(logging.INFO):
        trainer.train()

    assert any("Load pretrained SentenceTransformer" in text for _, _, text in caplog.record_tuples)


def test_evaluate_with_strings(model: SetFitModel) -> None:
    dataset = Dataset.from_dict({"text": ["a", "b", "c"], "label": ["positive", "positive", "negative"]})
    trainer = Trainer(model, train_dataset=dataset, eval_dataset=dataset)
    trainer.train()
    metrics = trainer.evaluate()
    assert "accuracy" in metrics


def test_trainer_wrong_args(model: SetFitModel, tmp_path: Path) -> None:
    args = TransformersTrainingArguments(output_dir=tmp_path)
    dataset = Dataset.from_dict({"text": ["a", "b", "c"], "label": [0, 1, 2]})
    expected = "`args` must be a `TrainingArguments` instance imported from `setfit`."
    with pytest.raises(ValueError, match=expected):
        Trainer(model, args=args)
    with pytest.raises(ValueError, match=expected):
        Trainer(model, dataset)
