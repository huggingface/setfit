from __future__ import annotations

import inspect
from copy import copy
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Dict, Tuple, Union

from sentence_transformers.losses.BatchHardTripletLoss import BatchHardTripletLossDistanceFunction


@dataclass
class TrainingArguments:

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

    embedding_learning_rate: float = 2e-5
    classifier_learning_rate: Union[float, Tuple[float, float]] = (1e-5, 1e-2)

    seed: int = 42
    use_amp: bool = False
    warmup_proportion: float = 0.1
    distance_metric: Callable = BatchHardTripletLossDistanceFunction.cosine_distance
    margin: float = 0.25
    samples_per_label: int = 2
    show_progress_bar: bool = True

    l2_weight: float = None
    max_length: int = None

    end_to_end: bool = False

    def __post_init__(self):
        if isinstance(self.batch_size, int):
            self.batch_size = (self.batch_size, self.batch_size)
        if self.embedding_batch_size is None:
            self.embedding_batch_size = self.batch_size[0]
        if self.classifier_batch_size is None:
            self.classifier_batch_size = self.batch_size[1]

        if isinstance(self.num_epochs, int):
            self.num_epochs = (self.num_epochs, self.num_epochs)
        if self.embedding_num_epochs is None:
            self.embedding_num_epochs = self.num_epochs[0]
        if self.classifier_num_epochs is None:
            self.classifier_num_epochs = self.num_epochs[1]

        if isinstance(self.classifier_learning_rate, float):
            self.classifier_learning_rate = (self.embedding_learning_rate, self.classifier_learning_rate)

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
