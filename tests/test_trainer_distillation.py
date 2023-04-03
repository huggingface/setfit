from unittest import TestCase

import pytest
from datasets import Dataset
from sentence_transformers.losses import CosineSimilarityLoss

from setfit import DistillationSetFitTrainer, SetFitTrainer
from setfit.modeling import SetFitModel


class DistillationSetFitTrainerTest(TestCase):
    def setUp(self):
        self.teacher_model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-albert-small-v2")
        self.student_model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-MiniLM-L3-v2")
        self.num_iterations = 1

    def test_trainer_works_with_default_columns(self):
        dataset = Dataset.from_dict({"text": ["a", "b", "c"], "label": [0, 1, 2], "extra_column": ["d", "e", "f"]})
        # train a teacher model
        teacher_trainer = SetFitTrainer(
            model=self.teacher_model,
            train_dataset=dataset,
            eval_dataset=dataset,
            loss_class=CosineSimilarityLoss,
            metric="accuracy",
        )
        # Teacher Train and evaluate
        teacher_trainer.train()
        metrics = teacher_trainer.evaluate()
        teacher_model = teacher_trainer.model

        student_trainer = DistillationSetFitTrainer(
            teacher_model=teacher_model,
            train_dataset=dataset,
            student_model=self.student_model,
            eval_dataset=dataset,
            loss_class=CosineSimilarityLoss,
            metric="accuracy",
        )

        # Student Train and evaluate
        student_trainer.train()
        metrics = student_trainer.evaluate()
        print("Student results: ", metrics)
        self.assertEqual(metrics["accuracy"], 1.0)

    def test_trainer_raises_error_with_missing_label(self):
        dataset = Dataset.from_dict({"text": ["a", "b", "c"], "extra_column": ["d", "e", "f"]})
        trainer = DistillationSetFitTrainer(
            teacher_model=self.teacher_model,
            train_dataset=dataset,
            student_model=self.student_model,
            eval_dataset=dataset,
            num_iterations=self.num_iterations,
        )
        with pytest.raises(ValueError):
            trainer.train()

    def test_trainer_raises_error_with_missing_text(self):
        dataset = Dataset.from_dict({"label": [0, 1, 2], "extra_column": ["d", "e", "f"]})
        trainer = DistillationSetFitTrainer(
            teacher_model=self.teacher_model,
            train_dataset=dataset,
            student_model=self.student_model,
            eval_dataset=dataset,
            num_iterations=self.num_iterations,
        )
        with pytest.raises(ValueError):
            trainer.train()

    def test_column_mapping_with_missing_text(self):
        dataset = Dataset.from_dict({"text": ["a", "b", "c"], "extra_column": ["d", "e", "f"]})
        trainer = DistillationSetFitTrainer(
            teacher_model=self.teacher_model,
            train_dataset=dataset,
            student_model=self.student_model,
            eval_dataset=dataset,
            num_iterations=self.num_iterations,
            column_mapping={"label_new": "label"},
        )
        with pytest.raises(ValueError):
            trainer._validate_column_mapping(trainer.train_dataset)

    def test_column_mapping_multilabel(self):
        dataset = Dataset.from_dict({"text_new": ["a", "b", "c"], "label_new": [[0, 1], [1, 2], [2, 0]]})

        trainer = DistillationSetFitTrainer(
            teacher_model=self.teacher_model,
            train_dataset=dataset,
            student_model=self.student_model,
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


def train_diff(trainer: SetFitTrainer):
    # Teacher Train and evaluate
    trainer.freeze()  # Freeze the head
    trainer.train()  # Train only the body

    # Unfreeze the head and unfreeze the body -> end-to-end training
    trainer.unfreeze(keep_body_frozen=False)

    trainer.train(num_epochs=5)


def train_lr(trainer: SetFitTrainer):
    trainer.train()


@pytest.mark.parametrize(("teacher_diff", "student_diff"), [[True, False], [True, False]])
def test_differentiable_models(teacher_diff: bool, student_diff: bool) -> None:
    if teacher_diff:
        teacher_model = SetFitModel.from_pretrained(
            "sentence-transformers/paraphrase-albert-small-v2",
            use_differentiable_head=True,
            head_params={"out_features": 3},
        )
        teacher_train_func = train_diff
    else:
        teacher_model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-albert-small-v2")
        teacher_train_func = train_lr
    if student_diff:
        student_model = SetFitModel.from_pretrained(
            "sentence-transformers/paraphrase-MiniLM-L3-v2",
            use_differentiable_head=True,
            head_params={"out_features": 3},
        )
        student_train_func = train_diff
    else:
        student_model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-MiniLM-L3-v2")
        student_train_func = train_lr

    dataset = Dataset.from_dict({"text": ["a", "b", "c"], "label": [0, 1, 2], "extra_column": ["d", "e", "f"]})
    # train a teacher model
    teacher_trainer = SetFitTrainer(
        model=teacher_model,
        train_dataset=dataset,
        eval_dataset=dataset,
        metric="accuracy",
    )
    teacher_train_func(teacher_trainer)
    metrics = teacher_trainer.evaluate()
    teacher_model = teacher_trainer.model

    student_trainer = DistillationSetFitTrainer(
        teacher_model=teacher_model,
        train_dataset=dataset,
        student_model=student_model,
        eval_dataset=dataset,
        metric="accuracy",
    )

    # Student Train and evaluate
    student_train_func(student_trainer)
    metrics = student_trainer.evaluate()
    print("Student results: ", metrics)
    assert metrics["accuracy"] == 1.0
