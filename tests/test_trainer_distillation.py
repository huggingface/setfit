from unittest import TestCase

import pytest
from datasets import Dataset

from setfit import DistillationTrainer, Trainer
from setfit.modeling import SetFitModel
from setfit.training_args import TrainingArguments


class DistillationTrainerTest(TestCase):
    def setUp(self):
        self.teacher_model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-albert-small-v2")
        self.student_model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-MiniLM-L3-v2")
        self.args = TrainingArguments(num_iterations=1)

    def test_trainer_works_with_default_columns(self):
        dataset = Dataset.from_dict({"text": ["a", "b", "c"], "label": [0, 1, 2], "extra_column": ["d", "e", "f"]})
        # train a teacher model
        teacher_trainer = Trainer(
            model=self.teacher_model,
            train_dataset=dataset,
            eval_dataset=dataset,
            metric="accuracy",
        )
        # Teacher Train and evaluate
        teacher_trainer.train()
        teacher_model = teacher_trainer.model

        student_trainer = DistillationTrainer(
            teacher_model=teacher_model,
            train_dataset=dataset,
            student_model=self.student_model,
            eval_dataset=dataset,
            metric="accuracy",
        )

        # Student Train and evaluate
        student_trainer.train()
        metrics = student_trainer.evaluate()
        print("Student results: ", metrics)
        self.assertEqual(metrics["accuracy"], 1.0)

    def test_trainer_raises_error_with_missing_label(self):
        labeled_dataset = Dataset.from_dict(
            {"text": ["a", "b", "c"], "label": [0, 1, 2], "extra_column": ["d", "e", "f"]}
        )
        # train a teacher model
        teacher_trainer = Trainer(
            model=self.teacher_model,
            train_dataset=labeled_dataset,
            eval_dataset=labeled_dataset,
            metric="accuracy",
            args=self.args,
        )
        # Teacher Train and evaluate
        teacher_trainer.train()

        unlabeled_dataset = Dataset.from_dict({"text": ["a", "b", "c"], "extra_column": ["d", "e", "f"]})
        student_trainer = DistillationTrainer(
            teacher_model=self.teacher_model,
            student_model=self.student_model,
            train_dataset=unlabeled_dataset,
            eval_dataset=labeled_dataset,
            args=self.args,
        )
        student_trainer.train()
        metrics = student_trainer.evaluate()
        print("Student results: ", metrics)
        self.assertEqual(metrics["accuracy"], 1.0)

    def test_trainer_raises_error_with_missing_text(self):
        dataset = Dataset.from_dict({"label": [0, 1, 2], "extra_column": ["d", "e", "f"]})
        with pytest.raises(ValueError):
            DistillationTrainer(
                teacher_model=self.teacher_model,
                train_dataset=dataset,
                student_model=self.student_model,
                eval_dataset=dataset,
                args=self.args,
            )

    def test_column_mapping_with_missing_text(self):
        dataset = Dataset.from_dict({"text": ["a", "b", "c"], "extra_column": ["d", "e", "f"]})
        with pytest.raises(ValueError):
            DistillationTrainer(
                teacher_model=self.teacher_model,
                train_dataset=dataset,
                student_model=self.student_model,
                eval_dataset=dataset,
                args=self.args,
                column_mapping={"label_new": "label"},
            )

    def test_column_mapping_multilabel(self):
        dataset = Dataset.from_dict({"text_new": ["a", "b", "c"], "label_new": [[0, 1], [1, 2], [2, 0]]})

        trainer = DistillationTrainer(
            teacher_model=self.teacher_model,
            train_dataset=dataset,
            student_model=self.student_model,
            eval_dataset=dataset,
            args=self.args,
            column_mapping={"text_new": "text", "label_new": "label"},
        )

        trainer._validate_column_mapping(dataset)
        formatted_dataset = trainer._apply_column_mapping(dataset, trainer.column_mapping)

        assert formatted_dataset.column_names == ["text", "label"]
        assert formatted_dataset[0]["text"] == "a"
        assert formatted_dataset[0]["label"] == [0, 1]
        assert formatted_dataset[1]["text"] == "b"


@pytest.mark.parametrize("teacher_diff", [True, False])
@pytest.mark.parametrize("student_diff", [True, False])
def test_differentiable_models(teacher_diff: bool, student_diff: bool) -> None:
    if teacher_diff:
        teacher_model = SetFitModel.from_pretrained(
            "sentence-transformers/paraphrase-albert-small-v2",
            use_differentiable_head=True,
            head_params={"out_features": 3},
        )
    else:
        teacher_model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-albert-small-v2")
    if student_diff:
        student_model = SetFitModel.from_pretrained(
            "sentence-transformers/paraphrase-MiniLM-L3-v2",
            use_differentiable_head=True,
            head_params={"out_features": 3},
        )
    else:
        student_model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-MiniLM-L3-v2")

    dataset = Dataset.from_dict({"text": ["a", "b", "c"], "label": [0, 1, 2], "extra_column": ["d", "e", "f"]})
    # train a teacher model
    teacher_trainer = Trainer(
        model=teacher_model,
        train_dataset=dataset,
        eval_dataset=dataset,
        metric="accuracy",
    )
    teacher_trainer.train()
    metrics = teacher_trainer.evaluate()
    print("Teacher results: ", metrics)
    assert metrics["accuracy"] == 1.0
    teacher_model = teacher_trainer.model

    student_trainer = DistillationTrainer(
        teacher_model=teacher_model,
        train_dataset=dataset,
        student_model=student_model,
        eval_dataset=dataset,
        metric="accuracy",
    )

    # Student Train and evaluate
    student_trainer.train()
    metrics = student_trainer.evaluate()
    print("Student results: ", metrics)
    assert metrics["accuracy"] == 1.0
