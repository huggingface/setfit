import warnings
from typing import TYPE_CHECKING, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
from datasets import Dataset
from sentence_transformers import InputExample, losses, util
from torch import nn
from torch.utils.data import DataLoader

from . import logging
from .sampler import ContrastiveDistillationDataset
from .trainer import Trainer
from .training_args import TrainingArguments


if TYPE_CHECKING:
    from .modeling import SetFitModel

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


class DistillationTrainer(Trainer):
    """Trainer to compress a SetFit model with knowledge distillation.

    Args:
        teacher_model (`SetFitModel`):
            The teacher model to mimic.
        student_model (`SetFitModel`, *optional*):
            The model to train. If not provided, a `model_init` must be passed.
        args (`TrainingArguments`, *optional*):
            The training arguments to use.
        train_dataset (`Dataset`):
            The training dataset.
        eval_dataset (`Dataset`, *optional*):
            The evaluation dataset.
        model_init (`Callable[[], SetFitModel]`, *optional*):
            A function that instantiates the model to be used. If provided, each call to
            [`~DistillationTrainer.train`] will start from a new instance of the model as given by this
            function when a `trial` is passed.
        metric (`str` or `Callable`, *optional*, defaults to `"accuracy"`):
            The metric to use for evaluation. If a string is provided, we treat it as the metric
            name and load it with default settings.
            If a callable is provided, it must take two arguments (`y_pred`, `y_test`).
        column_mapping (`Dict[str, str]`, *optional*):
            A mapping from the column names in the dataset to the column names expected by the model.
            The expected format is a dictionary with the following format:
            `{"text_column_name": "text", "label_column_name: "label"}`.
    """

    _REQUIRED_COLUMNS = {"text"}

    def __init__(
        self,
        teacher_model: "SetFitModel",
        student_model: Optional["SetFitModel"] = None,
        args: TrainingArguments = None,
        train_dataset: Optional["Dataset"] = None,
        eval_dataset: Optional["Dataset"] = None,
        model_init: Optional[Callable[[], "SetFitModel"]] = None,
        metric: Union[str, Callable[["Dataset", "Dataset"], Dict[str, float]]] = "accuracy",
        column_mapping: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(
            model=student_model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            model_init=model_init,
            metric=metric,
            column_mapping=column_mapping,
        )

        self.teacher_model = teacher_model
        self.student_model = self.model

    def dataset_to_parameters(self, dataset: Dataset) -> List[Iterable]:
        return [dataset["text"]]

    def get_dataloader(
        self,
        x: List[str],
        y: Optional[Union[List[int], List[List[int]]]],
        args: TrainingArguments,
        max_pairs: int = -1,
    ) -> Tuple[DataLoader, nn.Module, int, int]:
        x_embd_student = self.teacher_model.model_body.encode(
            x, convert_to_tensor=self.teacher_model.has_differentiable_head
        )
        cos_sim_matrix = util.cos_sim(x_embd_student, x_embd_student)

        input_data = [InputExample(texts=[text]) for text in x]
        data_sampler = ContrastiveDistillationDataset(
            input_data, cos_sim_matrix, args.num_iterations, args.sampling_strategy, max_pairs=max_pairs
        )
        batch_size = min(args.embedding_batch_size, len(data_sampler))
        dataloader = DataLoader(data_sampler, batch_size=batch_size, drop_last=False)
        loss = args.loss(self.model.model_body)
        return dataloader, loss, batch_size, len(data_sampler)

    def train_classifier(self, x_train: List[str], args: Optional[TrainingArguments] = None) -> None:
        """
        Method to perform the classifier phase: fitting the student classifier head.

        Args:
            x_train (`List[str]`): A list of training sentences.
            args (`TrainingArguments`, *optional*):
                Temporarily change the training arguments for this training call.
        """
        y_train = self.teacher_model.predict(x_train, as_numpy=not self.student_model.has_differentiable_head)
        return super().train_classifier(x_train, y_train, args)


class DistillationSetFitTrainer(DistillationTrainer):
    """
    `DistillationSetFitTrainer` has been deprecated and will be removed in v2.0.0 of SetFit.
    Please use `DistillationTrainer` instead.
    """

    def __init__(
        self,
        teacher_model: "SetFitModel",
        student_model: Optional["SetFitModel"] = None,
        train_dataset: Optional["Dataset"] = None,
        eval_dataset: Optional["Dataset"] = None,
        model_init: Optional[Callable[[], "SetFitModel"]] = None,
        metric: Union[str, Callable[["Dataset", "Dataset"], Dict[str, float]]] = "accuracy",
        loss_class: torch.nn.Module = losses.CosineSimilarityLoss,
        num_iterations: int = 20,
        num_epochs: int = 1,
        learning_rate: float = 2e-5,
        batch_size: int = 16,
        seed: int = 42,
        column_mapping: Optional[Dict[str, str]] = None,
        use_amp: bool = False,
        warmup_proportion: float = 0.1,
    ) -> None:
        warnings.warn(
            "`DistillationSetFitTrainer` has been deprecated and will be removed in v2.0.0 of SetFit. "
            "Please use `DistillationTrainer` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        args = TrainingArguments(
            num_iterations=num_iterations,
            num_epochs=num_epochs,
            body_learning_rate=learning_rate,
            head_learning_rate=learning_rate,
            batch_size=batch_size,
            seed=seed,
            use_amp=use_amp,
            warmup_proportion=warmup_proportion,
            loss=loss_class,
        )
        super().__init__(
            teacher_model=teacher_model,
            student_model=student_model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            model_init=model_init,
            metric=metric,
            column_mapping=column_mapping,
        )
