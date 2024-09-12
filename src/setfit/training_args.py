from __future__ import annotations

import inspect
import json
from copy import copy
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
from sentence_transformers import losses
from transformers import IntervalStrategy
from transformers.integrations import get_available_reporting_integrations
from transformers.training_args import default_logdir
from transformers.utils import is_torch_available

from . import logging


logger = logging.get_logger(__name__)


@dataclass
class TrainingArguments:
    """
    TrainingArguments is the subset of the arguments which relate to the training loop itself.
    Note that training with SetFit consists of two phases behind the scenes: **finetuning embeddings** and
    **training a classification head**. As a result, some of the training arguments can be tuples,
    where the two values are used for each of the two phases, respectively. The second value is often only
    used when training the model was loaded using `use_differentiable_head=True`.

    Parameters:
        output_dir (`str`, defaults to `"checkpoints"`):
            The output directory where the model predictions and checkpoints will be written.
        batch_size (`Union[int, Tuple[int, int]]`, defaults to `(16, 2)`):
            Set the batch sizes for the embedding and classifier training phases respectively,
            or set both if an integer is provided.
            Note that the batch size for the classifier is only used with a differentiable PyTorch head.
        num_epochs (`Union[int, Tuple[int, int]]`, defaults to `(1, 16)`):
            Set the number of epochs the embedding and classifier training phases respectively,
            or set both if an integer is provided.
            Note that the number of epochs for the classifier is only used with a differentiable PyTorch head.
        max_steps (`int`, defaults to `-1`):
            If set to a positive number, the total number of training steps to perform. Overrides `num_epochs`.
            The training may stop before reaching the set number of steps when all data is exhausted.
        sampling_strategy (`str`, defaults to `"oversampling"`):
            The sampling strategy of how to draw pairs in training. Possible values are:

                - `"oversampling"`: Draws even number of positive/ negative sentence pairs until every
                    sentence pair has been drawn.
                - `"undersampling"`: Draws the minimum number of positive/ negative sentence pairs until
                    every sentence pair in the minority class has been drawn.
                - `"unique"`: Draws every sentence pair combination (likely resulting in unbalanced
                    number of positive/ negative sentence pairs).

            The default is set to `"oversampling"`, ensuring all sentence pairs are drawn at least once.
            Alternatively, setting `num_iterations` will override this argument and determine the number
            of generated sentence pairs.
        num_iterations (`int`, *optional*):
            If not set the `sampling_strategy` will determine the number of sentence pairs to generate.
            This argument sets the number of iterations to generate sentence pairs for
            and provides compatability with Setfit <v1.0.0.
            This argument is ignored if triplet loss is used.
            It is only used in conjunction with `CosineSimilarityLoss`.
        body_learning_rate (`Union[float, Tuple[float, float]]`, defaults to `(2e-5, 1e-5)`):
            Set the learning rate for the `SentenceTransformer` body for the embedding and classifier
            training phases respectively, or set both if a float is provided.
            Note that the body learning rate for the classifier is only used with a differentiable PyTorch
            head *and* if `end_to_end=True`.
        head_learning_rate (`float`, defaults to `1e-2`):
            Set the learning rate for the head for the classifier training phase. Only used with a
            differentiable PyTorch head.
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
            runs, use the `model_init` argument to [`Trainer`] to instantiate the model if it has some
            randomly initialized parameters.
        report_to (`str` or `List[str]`, *optional*, defaults to `"all"`):
            The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,
            `"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. Use `"all"` to report to
            all integrations installed, `"none"` for no integrations.
        run_name (`str`, *optional*):
            A descriptor for the run. Typically used for [wandb](https://www.wandb.com/) and
            [mlflow](https://www.mlflow.org/) logging.
        logging_dir (`str`, *optional*):
            [TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to
            *runs/**CURRENT_DATETIME_HOSTNAME***.
        logging_strategy (`str` or [`~transformers.trainer_utils.IntervalStrategy`], *optional*, defaults to `"steps"`):
            The logging strategy to adopt during training. Possible values are:

                - `"no"`: No logging is done during training.
                - `"epoch"`: Logging is done at the end of each epoch.
                - `"steps"`: Logging is done every `logging_steps`.

        logging_first_step (`bool`, *optional*, defaults to `False`):
            Whether to log and evaluate the first `global_step` or not.
        logging_steps (`int`, defaults to 50):
            Number of update steps between two logs if `logging_strategy="steps"`.
        eval_strategy (`str` or [`~transformers.trainer_utils.IntervalStrategy`], *optional*, defaults to `"no"`):
            The evaluation strategy to adopt during training. Possible values are:

                - `"no"`: No evaluation is done during training.
                - `"steps"`: Evaluation is done (and logged) every `eval_steps`.
                - `"epoch"`: Evaluation is done at the end of each epoch.

        eval_steps (`int`, *optional*):
            Number of update steps between two evaluations if `eval_strategy="steps"`. Will default to the same
            value as `logging_steps` if not set.
        eval_delay (`float`, *optional*):
            Number of epochs or steps to wait for before the first evaluation can be performed, depending on the
            eval_strategy.
        eval_max_steps (`int`, defaults to `-1`):
            If set to a positive number, the total number of evaluation steps to perform. The evaluation may stop
            before reaching the set number of steps when all data is exhausted.

        save_strategy (`str` or [`~transformers.trainer_utils.IntervalStrategy`], *optional*, defaults to `"steps"`):
            The checkpoint save strategy to adopt during training. Possible values are:

                - `"no"`: No save is done during training.
                - `"epoch"`: Save is done at the end of each epoch.
                - `"steps"`: Save is done every `save_steps`.
        save_steps (`int`, *optional*, defaults to 500):
            Number of updates steps before two checkpoint saves if `save_strategy="steps"`.
        save_total_limit (`int`, *optional*, defaults to `1`):
            If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in
            `output_dir`. Note, the best model is always preserved if the `eval_strategy` is not `"no"`.
        load_best_model_at_end (`bool`, *optional*, defaults to `False`):
            Whether or not to load the best model found during training at the end of training.

            <Tip>

            When set to `True`, the parameters `save_strategy` needs to be the same as `eval_strategy`, and in
            the case it is "steps", `save_steps` must be a round multiple of `eval_steps`.

            </Tip>
    """

    output_dir: str = "checkpoints"

    # batch_size is only used to conveniently set `embedding_batch_size` and `classifier_batch_size`
    # which are used in practice
    batch_size: Union[int, Tuple[int, int]] = field(default=(16, 2), repr=False)

    # num_epochs is only used to conveniently set `embedding_num_epochs` and `classifier_num_epochs`
    # which are used in practice
    num_epochs: Union[int, Tuple[int, int]] = field(default=(1, 16), repr=False)

    max_steps: int = -1

    sampling_strategy: str = "oversampling"
    num_iterations: Optional[int] = None

    # As with batch_size and num_epochs, the first value in the tuple is the learning rate
    # for the embeddings step, while the second value is the learning rate for the classifier step.
    body_learning_rate: Union[float, Tuple[float, float]] = field(default=(2e-5, 1e-5), repr=False)
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

    # Logging & callbacks
    report_to: str = "all"
    run_name: Optional[str] = None
    logging_dir: Optional[str] = None
    logging_strategy: str = "steps"
    logging_first_step: bool = True
    logging_steps: int = 50

    eval_strategy: str = "no"
    evaluation_strategy: str = field(default="no", repr=False, init=False) # Softly deprecated
    eval_steps: Optional[int] = None
    eval_delay: int = 0
    eval_max_steps: int = -1

    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: Optional[int] = 1

    load_best_model_at_end: bool = False
    metric_for_best_model: str = field(default="embedding_loss", repr=False)
    greater_is_better: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        # Set `self.embedding_batch_size` and `self.classifier_batch_size` using values from `self.batch_size`
        if isinstance(self.batch_size, int):
            self.batch_size = (self.batch_size, self.batch_size)

        # Set `self.embedding_num_epochs` and `self.classifier_num_epochs` using values from `self.num_epochs`
        if isinstance(self.num_epochs, int):
            self.num_epochs = (self.num_epochs, self.num_epochs)

        # Set `self.body_embedding_learning_rate` and `self.body_classifier_learning_rate` using
        # values from `self.body_learning_rate`
        if isinstance(self.body_learning_rate, float):
            self.body_learning_rate = (self.body_learning_rate, self.body_learning_rate)

        if self.warmup_proportion < 0.0 or self.warmup_proportion > 1.0:
            raise ValueError(
                f"warmup_proportion must be greater than or equal to 0.0 and less than or equal to 1.0! But it was: {self.warmup_proportion}"
            )

        if self.report_to in (None, "all", ["all"]):
            self.report_to = get_available_reporting_integrations()
        elif self.report_to in ("none", ["none"]):
            self.report_to = []
        elif not isinstance(self.report_to, list):
            self.report_to = [self.report_to]

        if self.logging_dir is None:
            self.logging_dir = default_logdir()

        self.logging_strategy = IntervalStrategy(self.logging_strategy)
        if self.evaluation_strategy and not self.eval_strategy:
            logger.warning(
                "The `evaluation_strategy` argument is deprecated and will be removed in a future version. "
                "Please use `eval_strategy` instead."
            )
            self.eval_strategy = self.evaluation_strategy
        self.eval_strategy = IntervalStrategy(self.eval_strategy)

        if self.eval_steps is not None and self.eval_strategy == IntervalStrategy.NO:
            logger.info('Using `eval_strategy="steps"` as `eval_steps` is defined.')
            self.eval_strategy = IntervalStrategy.STEPS

        # eval_steps has to be defined and non-zero, fallbacks to logging_steps if the latter is non-zero
        if self.eval_strategy == IntervalStrategy.STEPS and (self.eval_steps is None or self.eval_steps == 0):
            if self.logging_steps > 0:
                self.eval_steps = self.logging_steps
            else:
                raise ValueError(
                    f"evaluation strategy {self.eval_strategy} requires either non-zero `eval_steps` or"
                    " `logging_steps`"
                )

        # Sanity checks for load_best_model_at_end: we require save and eval strategies to be compatible.
        if self.load_best_model_at_end:
            if self.eval_strategy != self.save_strategy:
                raise ValueError(
                    "`load_best_model_at_end` requires the save and eval strategy to match, but found\n- Evaluation "
                    f"strategy: {self.eval_strategy}\n- Save strategy: {self.save_strategy}"
                )
            if self.eval_strategy == IntervalStrategy.STEPS and self.save_steps % self.eval_steps != 0:
                raise ValueError(
                    "`load_best_model_at_end` requires the saving steps to be a round multiple of the evaluation "
                    f"steps, but found {self.save_steps}, which is not a round multiple of {self.eval_steps}."
                )

        # logging_steps must be non-zero for logging_strategy that is other than 'no'
        if self.logging_strategy == IntervalStrategy.STEPS and self.logging_steps == 0:
            raise ValueError(f"Logging strategy {self.logging_strategy} requires non-zero `logging_steps`")

    @property
    def embedding_batch_size(self) -> int:
        return self.batch_size[0]

    @property
    def classifier_batch_size(self) -> int:
        return self.batch_size[1]

    @property
    def embedding_num_epochs(self) -> int:
        return self.num_epochs[0]

    @property
    def classifier_num_epochs(self) -> int:
        return self.num_epochs[1]

    @property
    def body_embedding_learning_rate(self) -> float:
        return self.body_learning_rate[0]

    @property
    def body_classifier_learning_rate(self) -> float:
        return self.body_learning_rate[1]

    def to_dict(self) -> Dict[str, Any]:
        """Convert this instance to a dictionary.

        Returns:
            `Dict[str, Any]`: The dictionary variant of this dataclass.
        """
        return {field.name: getattr(self, field.name) for field in fields(self) if field.init}

    @classmethod
    def from_dict(cls, arguments: Dict[str, Any], ignore_extra: bool = False) -> TrainingArguments:
        """Initialize a TrainingArguments instance from a dictionary.

        Args:
            arguments (`Dict[str, Any]`): A dictionary of arguments.
            ignore_extra (`bool`, *optional*): Whether to ignore arguments that do not occur in the
                TrainingArguments __init__ signature. Defaults to False.

        Returns:
            `TrainingArguments`: The instantiated TrainingArguments instance.
        """
        if ignore_extra:
            return cls(**{key: value for key, value in arguments.items() if key in inspect.signature(cls).parameters})
        return cls(**arguments)

    def copy(self) -> TrainingArguments:
        """Create a shallow copy of this TrainingArguments instance."""
        return copy(self)

    def update(self, arguments: Dict[str, Any], ignore_extra: bool = False) -> TrainingArguments:
        return TrainingArguments.from_dict({**self.to_dict(), **arguments}, ignore_extra=ignore_extra)

    def to_json_string(self):
        # Serializes this instance to a JSON string.
        return json.dumps({key: str(value) for key, value in self.to_dict().items()}, indent=2)

    def to_sanitized_dict(self) -> Dict[str, Any]:
        # Sanitized serialization to use with TensorBoard’s hparams
        d = self.to_dict()
        d = {**d, **{"train_batch_size": self.embedding_batch_size, "eval_batch_size": self.embedding_batch_size}}

        valid_types = [bool, int, float, str]
        if is_torch_available():
            valid_types.append(torch.Tensor)

        return {k: v if type(v) in valid_types else str(v) for k, v in d.items()}
