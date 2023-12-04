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
        labeled_dataset = Dataset.from_dict(
            {"text": ["a", "b", "c"], "label": [0, 1, 2], "extra_column": ["d", "e", "f"]}
        )
        # train a teacher model
        teacher_trainer = SetFitTrainer(
            model=self.teacher_model,
            train_dataset=labeled_dataset,
            eval_dataset=labeled_dataset,
            metric="accuracy",
            num_iterations=self.num_iterations,
        )
        # Teacher Train and evaluate
        teacher_trainer.train()

        unlabeled_dataset = Dataset.from_dict({"text": ["a", "b", "c"], "extra_column": ["d", "e", "f"]})
        student_trainer = DistillationSetFitTrainer(
            teacher_model=self.teacher_model,
            student_model=self.student_model,
            train_dataset=unlabeled_dataset,
            eval_dataset=labeled_dataset,
            num_iterations=self.num_iterations,
        )
        student_trainer.train()
        metrics = student_trainer.evaluate()
        print("Student results: ", metrics)
        self.assertEqual(metrics["accuracy"], 1.0)

    def test_trainer_raises_error_with_missing_text(self):
        dataset = Dataset.from_dict({"label": [0, 1, 2], "extra_column": ["d", "e", "f"]})
        with pytest.raises(ValueError):
            DistillationSetFitTrainer(
                teacher_model=self.teacher_model,
                train_dataset=dataset,
                student_model=self.student_model,
                eval_dataset=dataset,
                num_iterations=self.num_iterations,
            )

    def test_column_mapping_with_missing_text(self):
        dataset = Dataset.from_dict({"text": ["a", "b", "c"], "extra_column": ["d", "e", "f"]})
        with pytest.raises(ValueError):
            DistillationSetFitTrainer(
                teacher_model=self.teacher_model,
                train_dataset=dataset,
                student_model=self.student_model,
                eval_dataset=dataset,
                num_iterations=self.num_iterations,
                column_mapping={"label_new": "label"},
            )

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

        trainer._validate_column_mapping(dataset)
        formatted_dataset = trainer._apply_column_mapping(dataset, trainer.column_mapping)

        assert formatted_dataset.column_names == ["text", "label"]
        assert formatted_dataset[0]["text"] == "a"
        assert formatted_dataset[0]["label"] == [0, 1]
        assert formatted_dataset[1]["text"] == "b"
