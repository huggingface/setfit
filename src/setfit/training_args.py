from __future__ import annotations

import inspect
from copy import copy
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Dict, Optional, Tuple, Union

from sentence_transformers import losses


@dataclass
class TrainingArguments:
    """
    TrainingArguments is the subset of the arguments which relate to the training loop itself.

    Parameters:
        batch_size (`Union[int, Tuple[int, int]]`, defaults to `(16, 2)`):
            Set the batch sizes for the embedding and classifier training phases respectively,
            or set both if an integer is provided.
            Note that the batch size for the classifier is only used with a differentiable PyTorch head.
        num_epochs (`Union[int, Tuple[int, int]]`, defaults to `(1, 16)`):
            Set the number of epochs the embedding and classifier training phases respectively,
            or set both if an integer is provided.
            Note that the number of epochs for the classifier is only used with a differentiable PyTorch head.
        num_iterations (`int`, defaults to `20`):
            The number of iterations to generate sentence pairs for.
            This argument is ignored if triplet loss is used.
            It is only used in conjunction with `CosineSimilarityLoss`.
        body_learning_rate (`Union[float, Tuple[float, float]]`, defaults to `(2e-5, 1e-5)`):
            Set the learning rate for the `SentenceTransformer` body for the embedding and classifier
            training phases respectively, or set both if a float is provided.
            Note that the body learning rate for the classifier is only used with a differentiable PyTorch
            head *and* if `end_to_end=True`.
        head_learning_rate (`float`, defaults to `1e-2`):
            Set the learning rate for the head for the classifier training phase.
        loss (`nn.Module`, defaults to `CosineSimilarityLoss`):
            The loss function to use for contrastive training of the embedding training phase.
        distance_metric (`Callable`, defaults to `BatchHardTripletLossDistanceFunction.cosine_distance`):
            Function that returns a distance between two embeddings.
            It is set for the triplet loss and ignored for `CosineSimilarityLoss` and `SupConLoss`.
        margin (`float`, defaults to `0.25`):
            Margin for the triplet loss.
            Negative samples should be at least margin further apart from the anchor than the positive.
            It is ignored for `CosineSimilarityLoss`, `BatchHardSoftMarginTripletLoss` and `SupConLoss`.
        end_to_end (`bool`, defaults to `False`):
            If True, train the entire model end-to-end during the classifier training phase.
            Otherwise, freeze the `SentenceTransformer` body and only train the head.
            Only used with a differentiable PyTorch head.
        use_amp (`bool`, defaults to `False`):
            Whether to use Automatic Mixed Precision (AMP) during the embedding training phase.
            Only for Pytorch >= 1.6.0
        warmup_proportion (`float`, defaults to `0.1`):
            Proportion of the warmup in the total training steps.
            Must be greater than or equal to 0.0 and less than or equal to 1.0.
        l2_weight (`float`, *optional*):
            Optional l2 weight for both the model body and head, passed to the `AdamW` optimizer in the
            classifier training phase if a differentiable PyTorch head is used.
        max_length (`int`, *optional*):
            The maximum token length a tokenizer can generate. If not provided, the maximum length for
            the `SentenceTransformer` body is used.
        samples_per_label (`int`, defaults to `2`): Number of consecutive, random and unique samples drawn per label.
            This is only relevant for triplet loss and ignored for `CosineSimilarityLoss`.
            Batch size should be a multiple of samples_per_label.
        show_progress_bar (`bool`, defaults to `True`):
            Whether to display a progress bar for the training epochs and iterations.
        seed (`int`, defaults to `42`):
            Random seed that will be set at the beginning of training. To ensure reproducibility across
            runs, use the [`~SetTrainer.model_init`] function to instantiate the model if it has some
            randomly initialized parameters.
    """

    # batch_size is only used to conveniently set `embedding_batch_size` and `classifier_batch_size`
    # which are used in practice
    batch_size: Union[int, Tuple[int, int]] = field(default=(16, 2), repr=False)
    embedding_batch_size: int = None
    classifier_batch_size: int = None

    # num_epochs is only used to conveniently set `embedding_num_epochs` and `classifier_num_epochs`
    # which are used in practice
    num_epochs: Union[int, Tuple[int, int]] = field(default=(1, 16), repr=False)
    embedding_num_epochs: int = None
    classifier_num_epochs: int = None

    num_iterations: int = 20

    # As with batch_size and num_epochs, the first value in the tuple is the learning rate
    # for the embeddings step, while the second value is the learning rate for the classifier step.
    body_learning_rate: Union[float, Tuple[float, float]] = field(default=(2e-5, 1e-5), repr=False)
    body_embedding_learning_rate: float = None
    body_classifier_learning_rate: float = None
    head_learning_rate: float = 1e-2

    # Loss-related arguments
    loss: Callable = losses.CosineSimilarityLoss
    distance_metric: Callable = losses.BatchHardTripletLossDistanceFunction.cosine_distance
    margin: float = 0.25

    end_to_end: bool = field(default=False)

    use_amp: bool = False
    warmup_proportion: float = 0.1
    l2_weight: Optional[float] = None
    max_length: Optional[int] = None
    samples_per_label: int = 2

    # Arguments that do not affect performance
    show_progress_bar: bool = True
    seed: int = 42

    def __post_init__(self):
        # Set `self.embedding_batch_size` and `self.classifier_batch_size` using values from `self.batch_size`
        if isinstance(self.batch_size, int):
            self.batch_size = (self.batch_size, self.batch_size)
        if self.embedding_batch_size is None:
            self.embedding_batch_size = self.batch_size[0]
        if self.classifier_batch_size is None:
            self.classifier_batch_size = self.batch_size[1]

        # Set `self.embedding_num_epochs` and `self.classifier_num_epochs` using values from `self.num_epochs`
        if isinstance(self.num_epochs, int):
            self.num_epochs = (self.num_epochs, self.num_epochs)
        if self.embedding_num_epochs is None:
            self.embedding_num_epochs = self.num_epochs[0]
        if self.classifier_num_epochs is None:
            self.classifier_num_epochs = self.num_epochs[1]

        # Set `self.body_embedding_learning_rate` and `self.body_classifier_learning_rate` using
        # values from `self.body_learning_rate`
        if isinstance(self.body_learning_rate, float):
            self.body_learning_rate = (self.body_learning_rate, self.body_learning_rate)
        if self.body_embedding_learning_rate is None:
            self.body_embedding_learning_rate = self.body_learning_rate[0]
        if self.body_classifier_learning_rate is None:
            self.body_classifier_learning_rate = self.body_learning_rate[1]

        if self.warmup_proportion < 0.0 or self.warmup_proportion > 1.0:
            raise ValueError(
                f"warmup_proportion must be greater than or equal to 0.0 and less than or equal to 1.0! But it was: {self.warmup_proportion}"
            )

    def to_dict(self):
        # filter out fields that are defined as field(init=False)
        return {field.name: getattr(self, field.name) for field in fields(self) if field.init}

    @classmethod
    def from_dict(cls, arguments: Dict[str, Any], ignore_extra: bool = False) -> TrainingArguments:
        if ignore_extra:
            return cls(**{key: value for key, value in arguments.items() if key in inspect.signature(cls).parameters})
        return cls(**arguments)

    def copy(self) -> TrainingArguments:
        return copy(self)

    def update(self, arguments: Dict[str, Any], ignore_extra: bool = False) -> TrainingArguments:
        return TrainingArguments.from_dict({**self.to_dict(), **arguments}, ignore_extra=ignore_extra)
