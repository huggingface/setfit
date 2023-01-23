from unittest import TestCase

import pytest

from setfit.training_args import TrainingArguments


class TestTrainingArguments(TestCase):
    def test_training_args_raises_error_with_wrong_warmup_proportion(self):
        # warmup_proportion must not be > 1.0
        with pytest.raises(ValueError):
            TrainingArguments(warmup_proportion=1.1)

        # warmup_proportion must not be < 0.0
        with pytest.raises(ValueError):
            TrainingArguments(warmup_proportion=-0.1)

    def test_training_args_batch_sizes(self):
        batch_size_A = 12
        batch_size_B = 4
        batch_size_C = 6

        args = TrainingArguments(batch_size=batch_size_A)
        assert args.batch_size == (batch_size_A, batch_size_A)
        assert args.embedding_batch_size == batch_size_A
        assert args.classifier_batch_size == batch_size_A

        args = TrainingArguments(batch_size=(batch_size_A, batch_size_B))
        assert args.batch_size == (batch_size_A, batch_size_B)
        assert args.embedding_batch_size == batch_size_A
        assert args.classifier_batch_size == batch_size_B

        args = TrainingArguments(batch_size=(batch_size_A, batch_size_B), embedding_batch_size=batch_size_C)
        assert args.batch_size == (batch_size_A, batch_size_B)
        assert args.embedding_batch_size == batch_size_C
        assert args.classifier_batch_size == batch_size_B

        args = TrainingArguments(batch_size=batch_size_A, embedding_batch_size=batch_size_C)
        assert args.batch_size == (batch_size_A, batch_size_A)
        assert args.embedding_batch_size == batch_size_C
        assert args.classifier_batch_size == batch_size_A

    def test_training_args_num_epochs(self):
        num_epochs_A = 12
        num_epochs_B = 4
        num_epochs_C = 6

        args = TrainingArguments(num_epochs=num_epochs_A)
        assert args.num_epochs == (num_epochs_A, num_epochs_A)
        assert args.embedding_num_epochs == num_epochs_A
        assert args.classifier_num_epochs == num_epochs_A

        args = TrainingArguments(num_epochs=(num_epochs_A, num_epochs_B))
        assert args.num_epochs == (num_epochs_A, num_epochs_B)
        assert args.embedding_num_epochs == num_epochs_A
        assert args.classifier_num_epochs == num_epochs_B

        args = TrainingArguments(num_epochs=(num_epochs_A, num_epochs_B), embedding_num_epochs=num_epochs_C)
        assert args.num_epochs == (num_epochs_A, num_epochs_B)
        assert args.embedding_num_epochs == num_epochs_C
        assert args.classifier_num_epochs == num_epochs_B

        args = TrainingArguments(num_epochs=num_epochs_A, embedding_num_epochs=num_epochs_C)
        assert args.num_epochs == (num_epochs_A, num_epochs_A)
        assert args.embedding_num_epochs == num_epochs_C
        assert args.classifier_num_epochs == num_epochs_A

    def test_training_args_learning_rates(self):
        learning_rate_A = 1e-2
        learning_rate_B = 1e-3
        learning_rate_C = 1e-4

        base = TrainingArguments()

        args = TrainingArguments(body_learning_rate=learning_rate_A)
        assert args.body_learning_rate == (learning_rate_A, learning_rate_A)
        assert args.body_embedding_learning_rate == learning_rate_A
        assert args.body_classifier_learning_rate == learning_rate_A
        assert args.head_learning_rate == base.head_learning_rate

        args = TrainingArguments(body_learning_rate=(learning_rate_A, learning_rate_B))
        assert args.body_learning_rate == (learning_rate_A, learning_rate_B)
        assert args.body_embedding_learning_rate == learning_rate_A
        assert args.body_classifier_learning_rate == learning_rate_B
        assert args.head_learning_rate == base.head_learning_rate

        args = TrainingArguments(
            body_learning_rate=(learning_rate_A, learning_rate_B), head_learning_rate=learning_rate_C
        )
        assert args.body_learning_rate == (learning_rate_A, learning_rate_B)
        assert args.body_embedding_learning_rate == learning_rate_A
        assert args.body_classifier_learning_rate == learning_rate_B
        assert args.head_learning_rate == learning_rate_C

        args = TrainingArguments(
            body_learning_rate=learning_rate_A,
            body_embedding_learning_rate=learning_rate_B,
            head_learning_rate=learning_rate_C,
        )
        # Perhaps not ideal, but body_learning_rate is never used directly:
        assert args.body_learning_rate == (learning_rate_A, learning_rate_A)
        assert args.body_embedding_learning_rate == learning_rate_B
        assert args.body_classifier_learning_rate == learning_rate_A
        assert args.head_learning_rate == learning_rate_C

        args = TrainingArguments(
            body_classifier_learning_rate=learning_rate_A,
            body_embedding_learning_rate=learning_rate_B,
            head_learning_rate=learning_rate_C,
        )
        assert args.body_embedding_learning_rate == learning_rate_B
        assert args.body_classifier_learning_rate == learning_rate_A
        assert args.head_learning_rate == learning_rate_C
