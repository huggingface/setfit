from unittest import TestCase

import pytest
from transformers import IntervalStrategy

from setfit.training_args import TrainingArguments


class TestTrainingArguments(TestCase):
    def test_raises_error_with_wrong_warmup_proportion(self):
        # warmup_proportion must not be > 1.0
        with pytest.raises(ValueError):
            TrainingArguments(warmup_proportion=1.1)

        # warmup_proportion must not be < 0.0
        with pytest.raises(ValueError):
            TrainingArguments(warmup_proportion=-0.1)

    def test_batch_sizes(self):
        batch_size_A = 12
        batch_size_B = 4

        args = TrainingArguments(batch_size=batch_size_A)
        self.assertEqual(args.batch_size, (batch_size_A, batch_size_A))
        self.assertEqual(args.embedding_batch_size, batch_size_A)
        self.assertEqual(args.classifier_batch_size, batch_size_A)

        args = TrainingArguments(batch_size=(batch_size_A, batch_size_B))
        self.assertEqual(args.batch_size, (batch_size_A, batch_size_B))
        self.assertEqual(args.embedding_batch_size, batch_size_A)
        self.assertEqual(args.classifier_batch_size, batch_size_B)

    def test_num_epochs(self):
        num_epochs_A = 12
        num_epochs_B = 4

        args = TrainingArguments(num_epochs=num_epochs_A)
        self.assertEqual(args.num_epochs, (num_epochs_A, num_epochs_A))
        self.assertEqual(args.embedding_num_epochs, num_epochs_A)
        self.assertEqual(args.classifier_num_epochs, num_epochs_A)

        args = TrainingArguments(num_epochs=(num_epochs_A, num_epochs_B))
        self.assertEqual(args.num_epochs, (num_epochs_A, num_epochs_B))
        self.assertEqual(args.embedding_num_epochs, num_epochs_A)
        self.assertEqual(args.classifier_num_epochs, num_epochs_B)

    def test_learning_rates(self):
        learning_rate_A = 1e-2
        learning_rate_B = 1e-3

        base = TrainingArguments()

        args = TrainingArguments(body_learning_rate=learning_rate_A)
        self.assertEqual(args.body_learning_rate, (learning_rate_A, learning_rate_A))
        self.assertEqual(args.body_embedding_learning_rate, learning_rate_A)
        self.assertEqual(args.body_classifier_learning_rate, learning_rate_A)
        self.assertEqual(args.head_learning_rate, base.head_learning_rate)

        args = TrainingArguments(body_learning_rate=(learning_rate_A, learning_rate_B))
        self.assertEqual(args.body_learning_rate, (learning_rate_A, learning_rate_B))
        self.assertEqual(args.body_embedding_learning_rate, learning_rate_A)
        self.assertEqual(args.body_classifier_learning_rate, learning_rate_B)
        self.assertEqual(args.head_learning_rate, base.head_learning_rate)

    def test_report_to(self):
        args = TrainingArguments(report_to="none")
        self.assertEqual(args.report_to, ["none"])
        args = TrainingArguments(report_to=["none"])
        self.assertEqual(args.report_to, ["none"])
        args = TrainingArguments(report_to="hello")
        self.assertEqual(args.report_to, ["hello"])

    def test_eval_steps_without_eval_strat(self):
        args = TrainingArguments(eval_steps=5)
        self.assertEqual(args.eval_strategy, IntervalStrategy.STEPS)

    def test_eval_strat_steps_without_eval_steps(self):
        args = TrainingArguments(eval_strategy="steps")
        self.assertEqual(args.eval_steps, args.logging_steps)
        with self.assertRaises(ValueError):
            TrainingArguments(eval_strategy="steps", logging_steps=0, logging_strategy="no")

    def test_load_best_model(self):
        with self.assertRaises(ValueError):
            TrainingArguments(load_best_model_at_end=True, eval_strategy="steps", save_strategy="epoch")
        with self.assertRaises(ValueError):
            TrainingArguments(
                load_best_model_at_end=True,
                eval_strategy="steps",
                save_strategy="steps",
                eval_steps=100,
                save_steps=50,
            )
        # No error: save_steps is a round multiple of eval_steps
        TrainingArguments(
            load_best_model_at_end=True,
            eval_strategy="steps",
            save_strategy="steps",
            eval_steps=50,
            save_steps=100,
        )

    def test_logging_steps_zero(self):
        with self.assertRaises(ValueError):
            TrainingArguments(logging_strategy="steps", logging_steps=0)
