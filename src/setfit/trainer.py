import math
import os
import shutil
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, Union

import evaluate
import torch
from datasets import Dataset, DatasetDict
from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.datasets import SentenceLabelDataset
from sentence_transformers.losses.BatchHardTripletLoss import BatchHardTripletLossDistanceFunction
from sentence_transformers.util import batch_to_device
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from transformers.integrations import WandbCallback, get_reporting_integration_callbacks
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    IntervalStrategy,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_utils import (
    HPSearchBackend,
    default_compute_objective,
    number_of_arguments,
    set_seed,
    speed_metrics,
)
from transformers.utils.import_utils import is_in_notebook

from setfit.model_card import ModelCardCallback

from . import logging
from .integrations import default_hp_search_backend, is_optuna_available, run_hp_search_optuna
from .losses import SupConLoss
from .sampler import ContrastiveDataset
from .training_args import TrainingArguments
from .utils import BestRun, default_hp_space_optuna


if TYPE_CHECKING:
    import optuna

    from .modeling import SetFitModel

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback


class ColumnMappingMixin:
    _REQUIRED_COLUMNS = {"text", "label"}

    def _validate_column_mapping(self, dataset: "Dataset") -> None:
        """
        Validates the provided column mapping against the dataset.
        """
        column_names = set(dataset.column_names)
        if self.column_mapping is None and not self._REQUIRED_COLUMNS.issubset(column_names):
            # Issue #226: load_dataset will automatically assign points to "train" if no split is specified
            if column_names == {"train"} and isinstance(dataset, DatasetDict):
                raise ValueError(
                    "SetFit expected a Dataset, but it got a DatasetDict with the split ['train']. "
                    "Did you mean to select the training split with dataset['train']?"
                )
            elif isinstance(dataset, DatasetDict):
                raise ValueError(
                    f"SetFit expected a Dataset, but it got a DatasetDict with the splits {sorted(column_names)}. "
                    "Did you mean to select one of these splits from the dataset?"
                )
            else:
                raise ValueError(
                    f"SetFit expected the dataset to have the columns {sorted(self._REQUIRED_COLUMNS)}, "
                    f"but only the columns {sorted(column_names)} were found. "
                    "Either make sure these columns are present, or specify which columns to use with column_mapping in Trainer."
                )
        if self.column_mapping is not None:
            missing_columns = set(self._REQUIRED_COLUMNS)
            # Remove columns that will be provided via the column mapping
            missing_columns -= set(self.column_mapping.values())
            # Remove columns that will be provided because they are in the dataset & not mapped away
            missing_columns -= set(dataset.column_names) - set(self.column_mapping.keys())
            if missing_columns:
                raise ValueError(
                    f"The following columns are missing from the column mapping: {missing_columns}. "
                    "Please provide a mapping for all required columns."
                )
            if not set(self.column_mapping.keys()).issubset(column_names):
                raise ValueError(
                    f"The column mapping expected the columns {sorted(self.column_mapping.keys())} in the dataset, "
                    f"but the dataset had the columns {sorted(column_names)}."
                )

    def _apply_column_mapping(self, dataset: "Dataset", column_mapping: Dict[str, str]) -> "Dataset":
        """
        Applies the provided column mapping to the dataset, renaming columns accordingly.
        Extra features not in the column mapping are prefixed with `"feat_"`.
        """
        dataset = dataset.rename_columns(
            {
                **column_mapping,
                **{
                    col: f"feat_{col}"
                    for col in dataset.column_names
                    if col not in column_mapping and col not in self._REQUIRED_COLUMNS
                },
            }
        )
        dset_format = dataset.format
        dataset = dataset.with_format(
            type=dset_format["type"],
            columns=dataset.column_names,
            output_all_columns=dset_format["output_all_columns"],
            **dset_format["format_kwargs"],
        )
        return dataset


class Trainer(ColumnMappingMixin):
    """Trainer to train a SetFit model.

    Args:
        model (`SetFitModel`, *optional*):
            The model to train. If not provided, a `model_init` must be passed.
        args (`TrainingArguments`, *optional*):
            The training arguments to use.
        train_dataset (`Dataset`):
            The training dataset.
        eval_dataset (`Dataset`, *optional*):
            The evaluation dataset.
        model_init (`Callable[[], SetFitModel]`, *optional*):
            A function that instantiates the model to be used. If provided, each call to
            [`Trainer.train`] will start from a new instance of the model as given by this
            function when a `trial` is passed.
        metric (`str` or `Callable`, *optional*, defaults to `"accuracy"`):
            The metric to use for evaluation. If a string is provided, we treat it as the metric
            name and load it with default settings. If a callable is provided, it must take two arguments
            (`y_pred`, `y_test`) and return a dictionary with metric keys to values.
        metric_kwargs (`Dict[str, Any]`, *optional*):
            Keyword arguments passed to the evaluation function if `metric` is an evaluation string like "f1".
            For example useful for providing an averaging strategy for computing f1 in a multi-label setting.
        callbacks (`List[`[`~transformers.TrainerCallback`]`]`, *optional*):
            A list of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main/en/main_classes/callback).
            If you want to remove one of the default callbacks used, use the [`Trainer.remove_callback`] method.
        column_mapping (`Dict[str, str]`, *optional*):
            A mapping from the column names in the dataset to the column names expected by the model.
            The expected format is a dictionary with the following format:
            `{"text_column_name": "text", "label_column_name: "label"}`.
    """

    def __init__(
        self,
        model: Optional["SetFitModel"] = None,
        args: Optional[TrainingArguments] = None,
        train_dataset: Optional["Dataset"] = None,
        eval_dataset: Optional["Dataset"] = None,
        model_init: Optional[Callable[[], "SetFitModel"]] = None,
        metric: Union[str, Callable[["Dataset", "Dataset"], Dict[str, float]]] = "accuracy",
        metric_kwargs: Optional[Dict[str, Any]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        column_mapping: Optional[Dict[str, str]] = None,
    ) -> None:
        if args is not None and not isinstance(args, TrainingArguments):
            raise ValueError("`args` must be a `TrainingArguments` instance imported from `setfit`.")
        self.args = args or TrainingArguments()
        self.column_mapping = column_mapping
        if train_dataset:
            self._validate_column_mapping(train_dataset)
            if self.column_mapping is not None:
                logger.info("Applying column mapping to the training dataset")
                train_dataset = self._apply_column_mapping(train_dataset, self.column_mapping)
        self.train_dataset = train_dataset

        if eval_dataset:
            self._validate_column_mapping(eval_dataset)
            if self.column_mapping is not None:
                logger.info("Applying column mapping to the evaluation dataset")
                eval_dataset = self._apply_column_mapping(eval_dataset, self.column_mapping)
        self.eval_dataset = eval_dataset

        self.model_init = model_init
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        self.logs_mapper = {}

        # Seed must be set before instantiating the model when using model_init.
        set_seed(12)

        if model is None:
            if model_init is not None:
                model = self.call_model_init()
            else:
                raise RuntimeError("`Trainer` requires either a `model` or `model_init` argument.")
        else:
            if model_init is not None:
                raise RuntimeError("`Trainer` requires either a `model` or `model_init` argument, but not both.")

        self.model = model
        self.hp_search_backend = None

        # Setup the callbacks
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        if WandbCallback in callbacks:
            # Set the W&B project via environment variables if it's not already set
            os.environ.setdefault("WANDB_PROJECT", "setfit")
        # TODO: Observe optimizer and scheduler by wrapping SentenceTransformer._get_scheduler
        self.callback_handler = CallbackHandler(callbacks, self.model, self.model.model_body.tokenizer, None, None)
        self.state = TrainerState()
        self.control = TrainerControl()
        self.add_callback(DEFAULT_PROGRESS_CALLBACK if self.args.show_progress_bar else PrinterCallback)
        self.control = self.callback_handler.on_init_end(self.args, self.state, self.control)

        # Add the callback for filling the model card data with hyperparameters
        # and evaluation results
        self.add_callback(ModelCardCallback(self))

        self.callback_handler.on_init_end(args, self.state, self.control)

    def add_callback(self, callback: Union[type, TrainerCallback]) -> None:
        """
        Add a callback to the current list of [`~transformers.TrainerCallback`].

        Args:
           callback (`type` or [`~transformers.TrainerCallback`]):
               A [`~transformers.TrainerCallback`] class or an instance of a [`~transformers.TrainerCallback`]. In the
               first case, will instantiate a member of that class.
        """
        self.callback_handler.add_callback(callback)

    def pop_callback(self, callback: Union[type, TrainerCallback]) -> TrainerCallback:
        """
        Remove a callback from the current list of [`~transformers.TrainerCallback`] and returns it.

        If the callback is not found, returns `None` (and no error is raised).

        Args:
           callback (`type` or [`~transformers.TrainerCallback`]):
               A [`~transformers.TrainerCallback`] class or an instance of a [`~transformers.TrainerCallback`]. In the
               first case, will pop the first member of that class found in the list of callbacks.

        Returns:
            [`~transformers.TrainerCallback`]: The callback removed, if found.
        """
        return self.callback_handler.pop_callback(callback)

    def remove_callback(self, callback: Union[type, TrainerCallback]) -> None:
        """
        Remove a callback from the current list of [`~transformers.TrainerCallback`].

        Args:
           callback (`type` or [`~transformers.TrainerCallback`]):
               A [`~transformers.TrainerCallback`] class or an instance of a [`~transformers.TrainerCallback`]. In the
               first case, will remove the first member of that class found in the list of callbacks.
        """
        self.callback_handler.remove_callback(callback)

    def apply_hyperparameters(self, params: Dict[str, Any], final_model: bool = False) -> None:
        """Applies a dictionary of hyperparameters to both the trainer and the model

        Args:
            params (`Dict[str, Any]`): The parameters, usually from `BestRun.hyperparameters`
            final_model (`bool`, *optional*, defaults to `False`): If `True`, replace the `model_init()` function with a fixed model based on the parameters.
        """

        if self.args is not None:
            self.args = self.args.update(params, ignore_extra=True)
        else:
            self.args = TrainingArguments.from_dict(params, ignore_extra=True)

        # Seed must be set before instantiating the model when using model_init.
        set_seed(self.args.seed)
        self.model = self.model_init(params)
        if final_model:
            self.model_init = None

    def _hp_search_setup(self, trial: Union["optuna.Trial", Dict[str, Any]]) -> None:
        """HP search setup code"""

        # Heavily inspired by transformers.Trainer._hp_search_setup
        if self.hp_search_backend is None or trial is None:
            return

        if isinstance(trial, Dict):  # For passing a Dict to train() -- mostly unused for now
            params = trial
        elif self.hp_search_backend == HPSearchBackend.OPTUNA:
            params = self.hp_space(trial)
        else:
            raise ValueError("Invalid trial parameter")

        logger.info(f"Trial: {params}")
        self.apply_hyperparameters(params, final_model=False)

    def call_model_init(self, params: Optional[Dict[str, Any]] = None) -> "SetFitModel":
        model_init_argcount = number_of_arguments(self.model_init)
        if model_init_argcount == 0:
            model = self.model_init()
        elif model_init_argcount == 1:
            model = self.model_init(params)
        else:
            raise RuntimeError("`model_init` should have 0 or 1 argument.")

        if model is None:
            raise RuntimeError("`model_init` should not return None.")

        return model

    def freeze(self, component: Optional[Literal["body", "head"]] = None) -> None:
        """Freeze the model body and/or the head, preventing further training on that component until unfrozen.

        This method is deprecated, use `SetFitModel.freeze` instead.

        Args:
            component (`Literal["body", "head"]`, *optional*): Either "body" or "head" to freeze that component.
                If no component is provided, freeze both. Defaults to None.
        """
        warnings.warn(
            f"`{self.__class__.__name__}.freeze` is deprecated and will be removed in v2.0.0 of SetFit. "
            "Please use `SetFitModel.freeze` directly instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.model.freeze(component)

    def unfreeze(
        self, component: Optional[Literal["body", "head"]] = None, keep_body_frozen: Optional[bool] = None
    ) -> None:
        """Unfreeze the model body and/or the head, allowing further training on that component.

        This method is deprecated, use `SetFitModel.unfreeze` instead.

        Args:
            component (`Literal["body", "head"]`, *optional*): Either "body" or "head" to unfreeze that component.
                If no component is provided, unfreeze both. Defaults to None.
            keep_body_frozen (`bool`, *optional*): Deprecated argument, use `component` instead.
        """
        warnings.warn(
            f"`{self.__class__.__name__}.unfreeze` is deprecated and will be removed in v2.0.0 of SetFit. "
            "Please use `SetFitModel.unfreeze` directly instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.model.unfreeze(component, keep_body_frozen=keep_body_frozen)

    def train(
        self,
        args: Optional[TrainingArguments] = None,
        trial: Optional[Union["optuna.Trial", Dict[str, Any]]] = None,
        **kwargs,
    ) -> None:
        """
        Main training entry point.

        Args:
            args (`TrainingArguments`, *optional*):
                Temporarily change the training arguments for this training call.
            trial (`optuna.Trial` or `Dict[str, Any]`, *optional*):
                The trial run or the hyperparameter dictionary for hyperparameter search.
        """
        if len(kwargs):
            warnings.warn(
                f"`{self.__class__.__name__}.train` does not accept keyword arguments anymore. "
                f"Please provide training arguments via a `TrainingArguments` instance to the `{self.__class__.__name__}` "
                f"initialisation or the `{self.__class__.__name__}.train` method.",
                DeprecationWarning,
                stacklevel=2,
            )

        if trial:  # Trial and model initialization
            self._hp_search_setup(trial)  # sets trainer parameters and initializes model

        args = args or self.args or TrainingArguments()

        if self.train_dataset is None:
            raise ValueError(
                f"Training requires a `train_dataset` given to the `{self.__class__.__name__}` initialization."
            )

        train_parameters = self.dataset_to_parameters(self.train_dataset)
        full_parameters = (
            train_parameters + self.dataset_to_parameters(self.eval_dataset) if self.eval_dataset else train_parameters
        )

        self.train_embeddings(*full_parameters, args=args)
        self.train_classifier(*train_parameters, args=args)

    def dataset_to_parameters(self, dataset: Dataset) -> List[Iterable]:
        return [dataset["text"], dataset["label"]]

    def train_embeddings(
        self,
        x_train: List[str],
        y_train: Optional[Union[List[int], List[List[int]]]] = None,
        x_eval: Optional[List[str]] = None,
        y_eval: Optional[Union[List[int], List[List[int]]]] = None,
        args: Optional[TrainingArguments] = None,
    ) -> None:
        """
        Method to perform the embedding phase: finetuning the `SentenceTransformer` body.

        Args:
            x_train (`List[str]`): A list of training sentences.
            y_train (`Union[List[int], List[List[int]]]`): A list of labels corresponding to the training sentences.
            args (`TrainingArguments`, *optional*):
                Temporarily change the training arguments for this training call.
        """
        args = args or self.args or TrainingArguments()
        # Since transformers v4.32.0, the log/eval/save steps should be saved on the state instead
        self.state.logging_steps = args.logging_steps
        self.state.eval_steps = args.eval_steps
        self.state.save_steps = args.save_steps
        # Reset the state
        self.state.global_step = 0
        self.state.total_flos = 0

        train_max_pairs = -1 if args.max_steps == -1 else args.max_steps * args.embedding_batch_size
        train_dataloader, loss_func, batch_size, num_unique_pairs = self.get_dataloader(
            x_train, y_train, args=args, max_pairs=train_max_pairs
        )
        if x_eval is not None and args.eval_strategy != IntervalStrategy.NO:
            eval_max_pairs = -1 if args.eval_max_steps == -1 else args.eval_max_steps * args.embedding_batch_size
            eval_dataloader, _, _, _ = self.get_dataloader(x_eval, y_eval, args=args, max_pairs=eval_max_pairs)
        else:
            eval_dataloader = None

        total_train_steps = len(train_dataloader) * args.embedding_num_epochs
        if args.max_steps > 0:
            total_train_steps = min(args.max_steps, total_train_steps)
        logger.info("***** Running training *****")
        logger.info(f"  Num unique pairs = {num_unique_pairs}")
        logger.info(f"  Batch size = {batch_size}")
        logger.info(f"  Num epochs = {args.embedding_num_epochs}")
        logger.info(f"  Total optimization steps = {total_train_steps}")

        warmup_steps = math.ceil(total_train_steps * args.warmup_proportion)
        self._train_sentence_transformer(
            self.model.model_body,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            args=args,
            loss_func=loss_func,
            warmup_steps=warmup_steps,
        )

    def get_dataloader(
        self, x: List[str], y: Union[List[int], List[List[int]]], args: TrainingArguments, max_pairs: int = -1
    ) -> Tuple[DataLoader, nn.Module, int, int]:
        # sentence-transformers adaptation
        input_data = [InputExample(texts=[text], label=label) for text, label in zip(x, y)]

        if args.loss in [
            losses.BatchAllTripletLoss,
            losses.BatchHardTripletLoss,
            losses.BatchSemiHardTripletLoss,
            losses.BatchHardSoftMarginTripletLoss,
            SupConLoss,
        ]:
            data_sampler = SentenceLabelDataset(input_data, samples_per_label=args.samples_per_label)
            batch_size = min(args.embedding_batch_size, len(data_sampler))
            dataloader = DataLoader(data_sampler, batch_size=batch_size, drop_last=True)

            if args.loss is losses.BatchHardSoftMarginTripletLoss:
                loss = args.loss(
                    model=self.model.model_body,
                    distance_metric=args.distance_metric,
                )
            elif args.loss is SupConLoss:
                loss = args.loss(model=self.model.model_body)
            else:
                loss = args.loss(
                    model=self.model.model_body,
                    distance_metric=args.distance_metric,
                    margin=args.margin,
                )
        else:
            data_sampler = ContrastiveDataset(
                input_data,
                self.model.multi_target_strategy,
                args.num_iterations,
                args.sampling_strategy,
                max_pairs=max_pairs,
            )
            batch_size = min(args.embedding_batch_size, len(data_sampler))
            dataloader = DataLoader(data_sampler, batch_size=batch_size, drop_last=False)
            loss = args.loss(self.model.model_body)

        return dataloader, loss, batch_size, len(data_sampler)

    def log(self, args: TrainingArguments, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        logs = {self.logs_mapper.get(key, key): value for key, value in logs.items()}
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        return self.callback_handler.on_log(args, self.state, self.control, logs)

    def _set_logs_mapper(self, logs_mapper: Dict[str, str]) -> None:
        """Set the logging mapper.

        Args:
            logs_mapper (str): The logging mapper, e.g. {"eval_embedding_loss": "eval_aspect_embedding_loss"}.
        """
        self.logs_mapper = logs_mapper

    def _train_sentence_transformer(
        self,
        model_body: SentenceTransformer,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader],
        args: TrainingArguments,
        loss_func: nn.Module,
        warmup_steps: int = 10000,
    ) -> None:
        """
        Train the model with the given training objective
        Each training objective is sampled in turn for one batch.
        We sample only as many batches from each objective as there are in the smallest one
        to make sure of equal training with each dataset.
        """
        # TODO: args.gradient_accumulation_steps
        # TODO: fp16/bf16, etc.
        # TODO: Safetensors

        # Hardcoded training arguments
        max_grad_norm = 1
        weight_decay = 0.01

        self.state.epoch = 0
        start_time = time.time()
        if args.max_steps > 0:
            self.state.max_steps = args.max_steps
        else:
            self.state.max_steps = len(train_dataloader) * args.embedding_num_epochs
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)
        steps_per_epoch = len(train_dataloader)

        if args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        model_body.to(self.model.device)
        loss_func.to(self.model.device)

        # Use smart batching
        train_dataloader.collate_fn = model_body.smart_batching_collate
        if eval_dataloader:
            eval_dataloader.collate_fn = model_body.smart_batching_collate

        # Prepare optimizers
        param_optimizer = list(loss_func.named_parameters())

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, **{"lr": args.body_embedding_learning_rate})
        scheduler_obj = model_body._get_scheduler(
            optimizer, scheduler="WarmupLinear", warmup_steps=warmup_steps, t_total=self.state.max_steps
        )
        self.callback_handler.optimizer = optimizer
        self.callback_handler.lr_scheduler = scheduler_obj
        self.callback_handler.train_dataloader = train_dataloader
        self.callback_handler.eval_dataloader = eval_dataloader

        self.callback_handler.on_train_begin(args, self.state, self.control)

        data_iterator = iter(train_dataloader)
        skip_scheduler = False
        for epoch in range(args.embedding_num_epochs):
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            loss_func.zero_grad()
            loss_func.train()

            for step in range(steps_per_epoch):
                self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                try:
                    data = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(train_dataloader)
                    data = next(data_iterator)

                features, labels = data
                labels = labels.to(self.model.device)
                features = list(map(lambda batch: batch_to_device(batch, self.model.device), features))

                if args.use_amp:
                    with autocast():
                        loss_value = loss_func(features, labels)

                    scale_before_step = scaler.get_scale()
                    scaler.scale(loss_value).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(loss_func.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()

                    skip_scheduler = scaler.get_scale() != scale_before_step
                else:
                    loss_value = loss_func(features, labels)
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(loss_func.parameters(), max_grad_norm)
                    optimizer.step()

                optimizer.zero_grad()

                if not skip_scheduler:
                    scheduler_obj.step()

                self.state.global_step += 1
                self.state.epoch = epoch + (step + 1) / steps_per_epoch
                self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                self.maybe_log_eval_save(model_body, eval_dataloader, args, scheduler_obj, loss_func, loss_value)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)

            self.maybe_log_eval_save(model_body, eval_dataloader, args, scheduler_obj, loss_func, loss_value)

            if self.control.should_training_stop:
                break

        if self.args.load_best_model_at_end and self.state.best_model_checkpoint:
            dir_name = Path(self.state.best_model_checkpoint).name
            if dir_name.startswith("step_"):
                step_to_load = dir_name[5:]
                logger.info(f"Loading best SentenceTransformer model from step {step_to_load}.")
                self.model.model_card_data.set_best_model_step(int(step_to_load))
            sentence_transformer_kwargs = self.model.sentence_transformers_kwargs
            sentence_transformer_kwargs["device"] = self.model.device
            self.model.model_body = SentenceTransformer(
                self.state.best_model_checkpoint, **sentence_transformer_kwargs
            )
            self.model.model_body.to(self.model.device)

        # Ensure logging the speed metrics
        num_train_samples = self.state.max_steps * args.embedding_batch_size  # * args.gradient_accumulation_steps
        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.control.should_log = True
        self.log(args, metrics)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

    def maybe_log_eval_save(
        self,
        model_body: SentenceTransformer,
        eval_dataloader: Optional[DataLoader],
        args: TrainingArguments,
        scheduler_obj,
        loss_func,
        loss_value: torch.Tensor,
    ) -> None:
        if self.control.should_log:
            learning_rate = scheduler_obj.get_last_lr()[0]
            metrics = {"embedding_loss": round(loss_value.item(), 4), "learning_rate": learning_rate}
            self.control = self.log(args, metrics)

        eval_loss = None
        if self.control.should_evaluate and eval_dataloader is not None:
            eval_loss = self._evaluate_with_loss(model_body, eval_dataloader, args, loss_func)
            learning_rate = scheduler_obj.get_last_lr()[0]
            metrics = {"eval_embedding_loss": round(eval_loss, 4), "learning_rate": learning_rate}
            self.control = self.log(args, metrics)

            self.control = self.callback_handler.on_evaluate(args, self.state, self.control, metrics)

            loss_func.zero_grad()
            loss_func.train()

        if self.control.should_save:
            checkpoint_dir = self._checkpoint(self.args.output_dir, args.save_total_limit, self.state.global_step)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

            if eval_loss is not None and (self.state.best_metric is None or eval_loss < self.state.best_metric):
                self.state.best_metric = eval_loss
                self.state.best_model_checkpoint = checkpoint_dir

    def _evaluate_with_loss(
        self,
        model_body: SentenceTransformer,
        eval_dataloader: DataLoader,
        args: TrainingArguments,
        loss_func: nn.Module,
    ) -> float:
        model_body.eval()
        losses = []
        eval_steps = (
            min(len(eval_dataloader), args.eval_max_steps) if args.eval_max_steps != -1 else len(eval_dataloader)
        )
        for step, data in enumerate(
            tqdm(iter(eval_dataloader), total=eval_steps, leave=False, disable=not args.show_progress_bar), start=1
        ):
            features, labels = data
            labels = labels.to(self.model.device)
            features = list(map(lambda batch: batch_to_device(batch, self.model.device), features))

            if args.use_amp:
                with autocast():
                    loss_value = loss_func(features, labels)

                losses.append(loss_value.item())
            else:
                losses.append(loss_func(features, labels).item())

            if step >= eval_steps:
                break

        model_body.train()
        return sum(losses) / len(losses)

    def _checkpoint(self, checkpoint_path: str, checkpoint_save_total_limit: int, step: int) -> None:
        # Delete old checkpoints
        if checkpoint_save_total_limit is not None and checkpoint_save_total_limit > 0:
            old_checkpoints = []
            for subdir in Path(checkpoint_path).glob("step_*"):
                if subdir.name[5:].isdigit() and (
                    self.state.best_model_checkpoint is None or subdir != Path(self.state.best_model_checkpoint)
                ):
                    old_checkpoints.append({"step": int(subdir.name[5:]), "path": str(subdir)})

            if len(old_checkpoints) > checkpoint_save_total_limit - 1:
                old_checkpoints = sorted(old_checkpoints, key=lambda x: x["step"])
                shutil.rmtree(old_checkpoints[0]["path"])

        checkpoint_file_path = str(Path(checkpoint_path) / f"step_{step}")
        self.model.save_pretrained(checkpoint_file_path)
        return checkpoint_file_path

    def train_classifier(
        self, x_train: List[str], y_train: Union[List[int], List[List[int]]], args: Optional[TrainingArguments] = None
    ) -> None:
        """
        Method to perform the classifier phase: fitting a classifier head.

        Args:
            x_train (`List[str]`): A list of training sentences.
            y_train (`Union[List[int], List[List[int]]]`): A list of labels corresponding to the training sentences.
            args (`TrainingArguments`, *optional*):
                Temporarily change the training arguments for this training call.
        """
        args = args or self.args or TrainingArguments()

        self.model.fit(
            x_train,
            y_train,
            num_epochs=args.classifier_num_epochs,
            batch_size=args.classifier_batch_size,
            body_learning_rate=args.body_classifier_learning_rate,
            head_learning_rate=args.head_learning_rate,
            l2_weight=args.l2_weight,
            max_length=args.max_length,
            show_progress_bar=args.show_progress_bar,
            end_to_end=args.end_to_end,
        )

    def evaluate(self, dataset: Optional[Dataset] = None, metric_key_prefix: str = "test") -> Dict[str, float]:
        """
        Computes the metrics for a given classifier.

        Args:
            dataset (`Dataset`, *optional*):
                The dataset to compute the metrics on. If not provided, will use the evaluation dataset passed via
                the `eval_dataset` argument at `Trainer` initialization.

        Returns:
            `Dict[str, float]`: The evaluation metrics.
        """

        if dataset is not None:
            self._validate_column_mapping(dataset)
            if self.column_mapping is not None:
                logger.info("Applying column mapping to the evaluation dataset")
                eval_dataset = self._apply_column_mapping(dataset, self.column_mapping)
            else:
                eval_dataset = dataset
        else:
            eval_dataset = self.eval_dataset

        if eval_dataset is None:
            raise ValueError("No evaluation dataset provided to `Trainer.evaluate` nor the `Trainer` initialzation.")

        x_test = eval_dataset["text"]
        y_test = eval_dataset["label"]

        logger.info("***** Running evaluation *****")
        y_pred = self.model.predict(x_test, use_labels=False)
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu()

        # Normalize string outputs
        if y_test and isinstance(y_test[0], str):
            encoder = LabelEncoder()
            encoder.fit(list(y_test) + list(y_pred))
            y_test = encoder.transform(y_test)
            y_pred = encoder.transform(y_pred)

        metric_kwargs = self.metric_kwargs or {}
        if isinstance(self.metric, str):
            metric_config = "multilabel" if self.model.multi_target_strategy is not None else None
            metric_fn = evaluate.load(self.metric, config_name=metric_config)

            results = metric_fn.compute(predictions=y_pred, references=y_test, **metric_kwargs)

        elif callable(self.metric):
            results = self.metric(y_pred, y_test, **metric_kwargs)

        else:
            raise ValueError("metric must be a string or a callable")

        if not isinstance(results, dict):
            results = {"metric": results}
        self.model.model_card_data.post_training_eval_results(
            {f"{metric_key_prefix}_{key}": value for key, value in results.items()}
        )
        return results

    def hyperparameter_search(
        self,
        hp_space: Optional[Callable[["optuna.Trial"], Dict[str, float]]] = None,
        compute_objective: Optional[Callable[[Dict[str, float]], float]] = None,
        n_trials: int = 10,
        direction: str = "maximize",
        backend: Optional[Union["str", HPSearchBackend]] = None,
        hp_name: Optional[Callable[["optuna.Trial"], str]] = None,
        **kwargs,
    ) -> BestRun:
        """
        Launch a hyperparameter search using `optuna`. The optimized quantity is determined
        by `compute_objective`, which defaults to a function returning the evaluation loss when no metric is provided,
        the sum of all metrics otherwise.

        <Tip warning={true}>

        To use this method, you need to have provided a `model_init` when initializing your [`Trainer`]: we need to
        reinitialize the model at each new run.

        </Tip>

        Args:
            hp_space (`Callable[["optuna.Trial"], Dict[str, float]]`, *optional*):
                A function that defines the hyperparameter search space. Will default to
                [`~transformers.trainer_utils.default_hp_space_optuna`].
            compute_objective (`Callable[[Dict[str, float]], float]`, *optional*):
                A function computing the objective to minimize or maximize from the metrics returned by the `evaluate`
                method. Will default to [`~transformers.trainer_utils.default_compute_objective`] which uses the sum of metrics.
            n_trials (`int`, *optional*, defaults to 100):
                The number of trial runs to test.
            direction (`str`, *optional*, defaults to `"maximize"`):
                Whether to optimize greater or lower objects. Can be `"minimize"` or `"maximize"`, you should pick
                `"minimize"` when optimizing the validation loss, `"maximize"` when optimizing one or several metrics.
            backend (`str` or [`~transformers.training_utils.HPSearchBackend`], *optional*):
                The backend to use for hyperparameter search. Only optuna is supported for now.
                TODO: add support for ray and sigopt.
            hp_name (`Callable[["optuna.Trial"], str]]`, *optional*):
                A function that defines the trial/run name. Will default to None.
            kwargs (`Dict[str, Any]`, *optional*):
                Additional keyword arguments passed along to `optuna.create_study`. For more
                information see:

                - the documentation of
                  [optuna.create_study](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.create_study.html)

        Returns:
            [`trainer_utils.BestRun`]: All the information about the best run.
        """
        if backend is None:
            backend = default_hp_search_backend()
            if backend is None:
                raise RuntimeError("optuna should be installed. To install optuna run `pip install optuna`.")
        backend = HPSearchBackend(backend)
        if backend == HPSearchBackend.OPTUNA and not is_optuna_available():
            raise RuntimeError("You picked the optuna backend, but it is not installed. Use `pip install optuna`.")
        elif backend != HPSearchBackend.OPTUNA:
            raise RuntimeError("Only optuna backend is supported for hyperparameter search.")
        self.hp_search_backend = backend
        if self.model_init is None:
            raise RuntimeError(
                "To use hyperparameter search, you need to pass your model through a model_init function."
            )

        self.hp_space = default_hp_space_optuna if hp_space is None else hp_space
        self.hp_name = hp_name
        self.compute_objective = default_compute_objective if compute_objective is None else compute_objective

        backend_dict = {
            HPSearchBackend.OPTUNA: run_hp_search_optuna,
        }
        best_run = backend_dict[backend](self, n_trials, direction, **kwargs)

        self.hp_search_backend = None
        return best_run

    def push_to_hub(self, repo_id: str, **kwargs) -> str:
        """Upload model checkpoint to the Hub using `huggingface_hub`.

        See the full list of parameters for your `huggingface_hub` version in the\
        [huggingface_hub documentation](https://huggingface.co/docs/huggingface_hub/package_reference/mixins#huggingface_hub.ModelHubMixin.push_to_hub).

        Args:
            repo_id (`str`):
                The full repository ID to push to, e.g. `"tomaarsen/setfit-sst2"`.
            config (`dict`, *optional*):
                Configuration object to be saved alongside the model weights.
            commit_message (`str`, *optional*):
                Message to commit while pushing.
            private (`bool`, *optional*, defaults to `False`):
                Whether the repository created should be private.
            api_endpoint (`str`, *optional*):
                The API endpoint to use when pushing the model to the hub.
            token (`str`, *optional*):
                The token to use as HTTP bearer authorization for remote files.
                If not set, will use the token set when logging in with
                `transformers-cli login` (stored in `~/.huggingface`).
            branch (`str`, *optional*):
                The git branch on which to push the model. This defaults to
                the default branch as specified in your repository, which
                defaults to `"main"`.
            create_pr (`boolean`, *optional*):
                Whether or not to create a Pull Request from `branch` with that commit.
                Defaults to `False`.
            allow_patterns (`List[str]` or `str`, *optional*):
                If provided, only files matching at least one pattern are pushed.
            ignore_patterns (`List[str]` or `str`, *optional*):
                If provided, files matching any of the patterns are not pushed.

        Returns:
            str: The url of the commit of your model in the given repository.
        """
        if "/" not in repo_id:
            raise ValueError(
                '`repo_id` must be a full repository ID, including organisation, e.g. "tomaarsen/setfit-sst2".'
            )
        commit_message = kwargs.pop("commit_message", "Add SetFit model")
        return self.model.push_to_hub(repo_id, commit_message=commit_message, **kwargs)


class SetFitTrainer(Trainer):
    """
    `SetFitTrainer` has been deprecated and will be removed in v2.0.0 of SetFit.
    Please use `Trainer` instead.
    """

    def __init__(
        self,
        model: Optional["SetFitModel"] = None,
        train_dataset: Optional["Dataset"] = None,
        eval_dataset: Optional["Dataset"] = None,
        model_init: Optional[Callable[[], "SetFitModel"]] = None,
        metric: Union[str, Callable[["Dataset", "Dataset"], Dict[str, float]]] = "accuracy",
        metric_kwargs: Optional[Dict[str, Any]] = None,
        loss_class=losses.CosineSimilarityLoss,
        num_iterations: int = 20,
        num_epochs: int = 1,
        learning_rate: float = 2e-5,
        batch_size: int = 16,
        seed: int = 42,
        column_mapping: Optional[Dict[str, str]] = None,
        use_amp: bool = False,
        warmup_proportion: float = 0.1,
        distance_metric: Callable = BatchHardTripletLossDistanceFunction.cosine_distance,
        margin: float = 0.25,
        samples_per_label: int = 2,
    ):
        warnings.warn(
            "`SetFitTrainer` has been deprecated and will be removed in v2.0.0 of SetFit. "
            "Please use `Trainer` instead.",
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
            distance_metric=distance_metric,
            margin=margin,
            samples_per_label=samples_per_label,
            loss=loss_class,
        )
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            model_init=model_init,
            metric=metric,
            metric_kwargs=metric_kwargs,
            column_mapping=column_mapping,
        )
