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