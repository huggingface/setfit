import math
import warnings
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from sentence_transformers import InputExample, losses, util
from sentence_transformers.datasets import SentenceLabelDataset
from torch.utils.data import DataLoader

from . import logging
from .losses import SupConLoss
from .modeling import sentence_pairs_generation_cos_sim
from .trainer import Trainer
from .training_args import TrainingArguments


if TYPE_CHECKING:
    from datasets import Dataset

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
            [`~SetFitTrainer.train`] will start from a new instance of the model as given by this
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

    def train_embeddings(
        self,
        x_train: List[str],
        y_train: Union[List[int], List[List[int]]],
        args: Optional[TrainingArguments] = None,
    ):
        args = args or self.args or TrainingArguments()

        # sentence-transformers adaptation
        if args.loss in [
            losses.BatchAllTripletLoss,
            losses.BatchHardTripletLoss,
            losses.BatchSemiHardTripletLoss,
            losses.BatchHardSoftMarginTripletLoss,
            SupConLoss,
        ]:
            train_examples = [InputExample(texts=[text], label=label) for text, label in zip(x_train, y_train)]
            train_data_sampler = SentenceLabelDataset(train_examples)

            batch_size = min(args.embedding_batch_size, len(train_data_sampler))
            train_dataloader = DataLoader(train_data_sampler, batch_size=batch_size, drop_last=True)

            if args.loss is losses.BatchHardSoftMarginTripletLoss:
                train_loss = args.loss(
                    model=self.student_model.model_body,
                    distance_metric=args.distance_metric,
                )
            elif args.loss is SupConLoss:
                train_loss = args.loss(model=self.student_model)
            else:
                train_loss = args.loss(
                    model=self.student_model.model_body,
                    distance_metric=args.distance_metric,
                    margin=args.margin,
                )
        else:
            train_examples = []

            # **************** student training *********************
            # Only this snippet differs from Trainer.train_embeddings
            x_train_embd_student = self.teacher_model.model_body.encode(x_train)
            y_train = self.teacher_model.model_head.predict(x_train_embd_student)

            cos_sim_matrix = util.cos_sim(x_train_embd_student, x_train_embd_student)

            train_examples = []
            for _ in range(args.num_iterations):
                train_examples = sentence_pairs_generation_cos_sim(np.array(x_train), train_examples, cos_sim_matrix)
            # **************** student training END *****************

            batch_size = args.embedding_batch_size
            train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
            train_loss = args.loss(self.student_model.model_body)

        total_train_steps = len(train_dataloader) * args.embedding_num_epochs
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_examples)}")
        logger.info(f"  Num epochs = {args.embedding_num_epochs}")
        logger.info(f"  Total optimization steps = {total_train_steps}")
        logger.info(f"  Total train batch size = {batch_size}")

        warmup_steps = math.ceil(total_train_steps * args.warmup_proportion)
        self.student_model.model_body.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=args.embedding_num_epochs,
            optimizer_params={"lr": args.body_embedding_learning_rate},
            warmup_steps=warmup_steps,
            show_progress_bar=args.show_progress_bar,
            use_amp=args.use_amp,
        )


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
    ):
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
