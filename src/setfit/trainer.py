import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, Union

import evaluate
import torch
from datasets import Dataset, DatasetDict
from packaging.version import parse as parse_version
from sentence_transformers import SentenceTransformerTrainer, losses
from sentence_transformers.losses.BatchHardTripletLoss import BatchHardTripletLossDistanceFunction
from sentence_transformers.model_card import ModelCardCallback as STModelCardCallback
from sentence_transformers.training_args import BatchSamplers, SentenceTransformerTrainingArguments
from sklearn.preprocessing import LabelEncoder
from torch import nn
from transformers import __version__ as transformers_version
from transformers.integrations import CodeCarbonCallback
from transformers.trainer_callback import IntervalStrategy, TrainerCallback
from transformers.trainer_utils import HPSearchBackend, default_compute_objective, number_of_arguments, set_seed
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


class BCSentenceTransformersTrainer(SentenceTransformerTrainer):
    """
    Subclass of SentenceTransformerTrainer that is backwards compatible with the SetFit API.
    """

    def __init__(
        self, setfit_model: "SetFitModel", setfit_args: "TrainingArguments", callbacks: List[TrainerCallback], **kwargs
    ):
        self._setfit_model = setfit_model
        self._setfit_args = setfit_args
        self.logs_prefix = "embedding"
        super().__init__(
            model=setfit_model.model_body,
            args=SentenceTransformerTrainingArguments(output_dir=setfit_args.output_dir),
            **kwargs,
        )
        self._apply_training_arguments(setfit_args)

        for callback in list(self.callback_handler.callbacks):
            if isinstance(callback, CodeCarbonCallback):
                self.setfit_model.model_card_data.code_carbon_callback = callback

            if isinstance(callback, STModelCardCallback):
                self.remove_callback(callback)

        if is_in_notebook():
            from transformers.utils.notebook import NotebookProgressCallback

            from setfit.notebook import SetFitNotebookProgressCallback

            if self.pop_callback(NotebookProgressCallback):
                self.add_callback(SetFitNotebookProgressCallback)

        def overwritten_call_event(self, event, args, state, control, **kwargs):
            for callback in self.callbacks:
                result = getattr(callback, event)(
                    self.setfit_args,
                    state,
                    control,
                    model=self.setfit_model,
                    st_model=self.model,
                    st_args=args,
                    tokenizer=(
                        self.processing_class
                        if parse_version(transformers_version) >= parse_version("4.46.0")
                        else self.tokenizer
                    ),
                    optimizer=self.optimizer,
                    lr_scheduler=self.lr_scheduler,
                    train_dataloader=self.train_dataloader,
                    eval_dataloader=self.eval_dataloader,
                    **kwargs,
                )
                # A Callback can skip the return of `control` if it doesn't change it.
                if result is not None:
                    control = result
            return control

        self.callback_handler.call_event = lambda *args, **kwargs: overwritten_call_event(
            self.callback_handler, *args, **kwargs
        )
        self.callback_handler.setfit_args = setfit_args
        self.callback_handler.setfit_model = setfit_model
        for callback in callbacks:
            self.add_callback(callback)
        self.callback_handler.on_init_end(self.args, self.state, self.control)

    def add_model_card_callback(self, *args, **kwargs):
        pass

    @property
    def setfit_model(self) -> "SetFitModel":
        return self._setfit_model

    @setfit_model.setter
    def setfit_model(self, setfit_model: "SetFitModel") -> None:
        self._setfit_model = setfit_model
        self.model = setfit_model.model_body
        self.callback_handler.setfit_model = setfit_model

    @property
    def setfit_args(self) -> TrainingArguments:
        return self._setfit_args

    @setfit_args.setter
    def setfit_args(self, setfit_args: TrainingArguments) -> None:
        self._setfit_args = setfit_args
        self._apply_training_arguments(setfit_args)
        self.callback_handler.setfit_args = setfit_args

    def _apply_training_arguments(self, args: TrainingArguments) -> None:
        """
        Propagate the SetFit TrainingArguments to the SentenceTransformer Trainer.
        """
        self.args.output_dir = args.output_dir

        self.args.per_device_train_batch_size = args.embedding_batch_size
        self.args.per_device_eval_batch_size = args.embedding_batch_size
        self.args.num_train_epochs = args.embedding_num_epochs
        self.args.max_steps = args.max_steps
        self.args.learning_rate = args.body_embedding_learning_rate
        self.args.fp16 = args.use_amp
        self.args.weight_decay = args.l2_weight
        self.args.seed = args.seed
        self.args.report_to = args.report_to
        self.args.run_name = args.run_name
        self.args.warmup_ratio = args.warmup_proportion

        self.args.logging_dir = args.logging_dir
        self.args.logging_strategy = args.logging_strategy
        self.args.logging_first_step = args.logging_first_step
        self.args.logging_steps = args.logging_steps

        self.args.eval_strategy = args.eval_strategy
        self.args.eval_steps = args.eval_steps
        self.args.eval_delay = args.eval_delay

        self.args.save_strategy = args.save_strategy
        self.args.save_steps = args.save_steps
        self.args.save_total_limit = args.save_total_limit

        self.args.load_best_model_at_end = args.load_best_model_at_end
        self.args.metric_for_best_model = args.metric_for_best_model
        self.args.greater_is_better = args.greater_is_better

    def _set_logs_prefix(self, logs_prefix: str) -> None:
        """Set the logging prefix.

        Args:
            logs_mapper (str): The logging prefix, e.g. "aspect_embedding".
        """
        self.logs_prefix = logs_prefix

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        logs = {f"{self.logs_prefix}_{k}" if k == "loss" else k: v for k, v in logs.items()}
        return super().log(logs, *args, **kwargs)

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        return super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=f"{metric_key_prefix}_{self.logs_prefix}",
        )


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
        set_seed(args.seed if args is not None else 12)

        if model is None:
            if model_init is not None:
                model = self.call_model_init()
            else:
                raise RuntimeError("`Trainer` requires either a `model` or `model_init` argument.")
        else:
            if model_init is not None:
                raise RuntimeError("`Trainer` requires either a `model` or `model_init` argument, but not both.")

        self._model = model
        self.hp_search_backend = None

        callbacks = callbacks + [ModelCardCallback(self)] if callbacks else [ModelCardCallback(self)]
        self.st_trainer = BCSentenceTransformersTrainer(
            setfit_model=model,
            setfit_args=self.args,
            callbacks=callbacks,
        )

    @property
    def args(self) -> TrainingArguments:
        return self._args

    @args.setter
    def args(self, args: TrainingArguments) -> None:
        self._args = args
        if hasattr(self, "st_trainer"):
            self.st_trainer.setfit_args = args

    @property
    def model(self) -> "SetFitModel":
        return self._model

    @model.setter
    def model(self, model: "SetFitModel") -> None:
        self._model = model
        if hasattr(self, "st_trainer"):
            self.st_trainer.setfit_model = model

    def add_callback(self, callback: Union[type, TrainerCallback]) -> None:
        """
        Add a callback to the current list of [`~transformers.TrainerCallback`].

        Args:
           callback (`type` or [`~transformers.TrainerCallback`]):
               A [`~transformers.TrainerCallback`] class or an instance of a [`~transformers.TrainerCallback`]. In the
               first case, will instantiate a member of that class.
        """
        self.st_trainer.add_callback(callback)

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
        return self.st_trainer.pop_callback(callback)

    def remove_callback(self, callback: Union[type, TrainerCallback]) -> None:
        """
        Remove a callback from the current list of [`~transformers.TrainerCallback`].

        Args:
           callback (`type` or [`~transformers.TrainerCallback`]):
               A [`~transformers.TrainerCallback`] class or an instance of a [`~transformers.TrainerCallback`]. In the
               first case, will remove the first member of that class found in the list of callbacks.
        """
        self.st_trainer.remove_callback(callback)

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
        if args:
            self.st_trainer.setfit_args = args
        args = args or self.args or TrainingArguments()

        train_max_pairs = -1 if args.max_steps == -1 else args.max_steps * args.embedding_batch_size
        train_dataset, loss = self.get_dataset(x_train, y_train, args=args, max_pairs=train_max_pairs)
        if x_eval is not None and args.eval_strategy != IntervalStrategy.NO:
            eval_max_pairs = -1 if args.eval_max_steps == -1 else args.eval_max_steps * args.embedding_batch_size
            eval_dataset, _ = self.get_dataset(x_eval, y_eval, args=args, max_pairs=eval_max_pairs)
        else:
            eval_dataset = None

        logger.info("***** Running training *****")
        logger.info(f"  Num unique pairs = {len(train_dataset)}")
        logger.info(f"  Batch size = {args.embedding_batch_size}")
        logger.info(f"  Num epochs = {args.embedding_num_epochs}")

        self.st_trainer.train_dataset = train_dataset
        self.st_trainer.eval_dataset = eval_dataset
        self.st_trainer.loss = loss
        if loss in (
            losses.BatchAllTripletLoss,
            losses.BatchHardTripletLoss,
            losses.BatchSemiHardTripletLoss,
            losses.BatchHardSoftMarginTripletLoss,
            SupConLoss,
        ):
            self.st_trainer.args.batch_sampler = BatchSamplers.GROUP_BY_LABEL
        self.st_trainer.train()

    def get_dataset(
        self, x: List[str], y: Union[List[int], List[List[int]]], args: TrainingArguments, max_pairs: int = -1
    ) -> Tuple[Dataset, nn.Module, int, int]:
        if args.loss in [
            losses.BatchAllTripletLoss,
            losses.BatchHardTripletLoss,
            losses.BatchSemiHardTripletLoss,
            losses.BatchHardSoftMarginTripletLoss,
            SupConLoss,
        ]:
            dataset = Dataset.from_dict({"sentence": x, "label": y})

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
                x,
                y,
                self.model.multi_target_strategy,
                args.num_iterations,
                args.sampling_strategy,
                max_pairs=max_pairs,
            )
            dataset = Dataset.from_list(list(data_sampler))
            loss = args.loss(self.model.model_body)

        return dataset, loss

    def _set_logs_prefix(self, logs_prefix: str) -> None:
        """Set the logging prefix.

        Args:
            logs_mapper (str): The logging prefix, e.g. "aspect_embedding".
        """
        self.st_trainer._set_logs_prefix(logs_prefix)

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
            private (`bool`, *optional*):
                Whether to make the repo private. If `None` (default), the repo will be public unless the organization's default is private. This value is ignored if the repo already exists.
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
