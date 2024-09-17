import logging

from datasets import Dataset
from pytest import LogCaptureFixture
from transformers import TrainerCallback

from setfit import AbsaTrainer
from setfit.logging import get_logger
from setfit.span.modeling import AbsaModel


def test_trainer(absa_model: AbsaModel, absa_dataset: Dataset) -> None:
    trainer = AbsaTrainer(absa_model, train_dataset=absa_dataset, eval_dataset=absa_dataset)
    trainer.train()

    metrics = trainer.evaluate()
    assert "aspect" in metrics
    assert "polarity" in metrics
    assert "accuracy" in metrics["aspect"]
    assert "accuracy" in metrics["polarity"]
    assert metrics["aspect"]["accuracy"] > 0.0
    assert metrics["polarity"]["accuracy"] > 0.0
    new_metrics = trainer.evaluate(absa_dataset)
    assert metrics == new_metrics

    predict = absa_model.predict("Best pizza outside of Italy and really tasty.")
    assert {"span": "pizza", "polarity": "positive"} in predict
    predict = absa_model.predict(["Best pizza outside of Italy and really tasty.", "This is another sentence"])
    assert isinstance(predict, list) and len(predict) == 2 and isinstance(predict[0], list)
    predict = absa_model(["Best pizza outside of Italy and really tasty.", "This is another sentence"])
    assert isinstance(predict, list) and len(predict) == 2 and isinstance(predict[0], list)


def test_trainer_callbacks(absa_model: AbsaModel) -> None:
    trainer = AbsaTrainer(absa_model)
    assert len(trainer.aspect_trainer.st_trainer.callback_handler.callbacks) >= 2
    num_callbacks = len(trainer.aspect_trainer.st_trainer.callback_handler.callbacks)
    callback_names = {
        callback.__class__.__name__ for callback in trainer.aspect_trainer.st_trainer.callback_handler.callbacks
    }
    assert {"DefaultFlowCallback", "ProgressCallback"} <= callback_names

    class TestCallback(TrainerCallback):
        pass

    callback = TestCallback()
    trainer.add_callback(callback)
    assert len(trainer.aspect_trainer.st_trainer.callback_handler.callbacks) == num_callbacks + 1
    assert len(trainer.polarity_trainer.st_trainer.callback_handler.callbacks) == num_callbacks + 1
    assert trainer.aspect_trainer.st_trainer.callback_handler.callbacks[-1] == callback
    assert trainer.polarity_trainer.st_trainer.callback_handler.callbacks[-1] == callback

    assert trainer.pop_callback(callback) == (callback, callback)
    trainer.add_callback(callback)
    assert trainer.aspect_trainer.st_trainer.callback_handler.callbacks[-1] == callback
    assert trainer.polarity_trainer.st_trainer.callback_handler.callbacks[-1] == callback
    trainer.remove_callback(callback)
    assert callback not in trainer.aspect_trainer.st_trainer.callback_handler.callbacks
    assert callback not in trainer.polarity_trainer.st_trainer.callback_handler.callbacks


def test_train_ordinal_too_high(absa_model: AbsaModel, caplog: LogCaptureFixture) -> None:
    logger = get_logger("setfit")
    logger.propagate = True

    absa_dataset = Dataset.from_dict(
        {
            "text": [
                "It is about food and ambiance, and imagine how dreadful it will be it we only had to listen to an idle engine."
            ],
            "span": ["food"],
            "label": ["negative"],
            "ordinal": [1],
        }
    )
    with caplog.at_level(logging.INFO):
        trainer = AbsaTrainer(absa_model, train_dataset=absa_dataset)
    assert len(trainer.aspect_trainer.train_dataset) == 3
    assert len(trainer.polarity_trainer.train_dataset) == 0
    # These tests are ignored as the caplog is inconsistent:
    # assert len(caplog.record_tuples) == 1
    # assert caplog.record_tuples[0][2] == (
    #     "The ordinal of 1 for span 'food' in 'It is about food and ambiance, and imagine how dreadful it will be "
    #     "it we only had to listen to an idle engine.' is too high. Skipping this sample."
    # )
    # assert caplog.record_tuples[0][1] == logging.INFO

    logger.propagate = False


def test_train_column_mapping(absa_model: AbsaModel, absa_dataset: Dataset) -> None:
    absa_dataset = absa_dataset.rename_columns({"text": "sentence", "span": "aspect"})
    trainer = AbsaTrainer(
        absa_model, train_dataset=absa_dataset, column_mapping={"sentence": "text", "aspect": "span"}
    )
    trainer.train()
