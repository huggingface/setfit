__version__ = "0.6.0.dev0"

from .data import add_templated_examples, sample_dataset
from .modeling import SetFitHead, SetFitModel
from .trainer import SetFitTrainer
from .trainer_distillation import DistillationSetFitTrainer
