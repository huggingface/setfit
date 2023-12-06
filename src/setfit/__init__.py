__version__ = "1.1.0.dev0"

import importlib
import os
import warnings

from .data import get_templated_dataset, sample_dataset
from .model_card import SetFitModelCardData
from .modeling import SetFitHead, SetFitModel
from .span import AbsaModel, AbsaTrainer, AspectExtractor, AspectModel, PolarityModel
from .trainer import SetFitTrainer, Trainer
from .trainer_distillation import DistillationSetFitTrainer, DistillationTrainer
from .training_args import TrainingArguments


# Ensure that DeprecationWarnings are shown by default, as recommended by
# https://docs.python.org/3/library/warnings.html#overriding-the-default-filter
warnings.filterwarnings("default", category=DeprecationWarning)

# If codecarbon is installed and the log level is not defined,
# automatically overwrite the default to "error"
if importlib.util.find_spec("codecarbon") and "CODECARBON_LOG_LEVEL" not in os.environ:
    os.environ["CODECARBON_LOG_LEVEL"] = "error"
