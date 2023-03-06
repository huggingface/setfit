import math
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union


# Google Colab runs on Python 3.7, so we need this to be compatible
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import evaluate
import numpy as np
from datasets import DatasetDict
from sentence_transformers import InputExample, losses
from sentence_transformers.datasets import SentenceLabelDataset
from sentence_transformers.losses.BatchHardTripletLoss import BatchHardTripletLossDistanceFunction
from torch.utils.data import DataLoader
from tqdm.auto import trange
from transformers.trainer_utils import HPSearchBackend, default_compute_objective, number_of_arguments, set_seed

from . import logging
from .integrations import default_hp_search_backend, is_optuna_available, run_hp_search_optuna
from .losses import SupConLoss
from .modeling import sentence_pairs_generation, sentence_pairs_generation_multilabel
from .training_args import TrainingArguments
from .utils import BestRun, default_hp_space_optuna


if TYPE_CHECKING:
    import optuna
    from datasets import Dataset

    from .modeling import SetFitModel

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


class Trainer:
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
            [`~Trainer.train`] will start from a new instance of the model as given by this
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

    _REQUIRED_COLUMNS = {"text", "label"}

    def __init__(
        self,
        model: Optional["SetFitModel"] = None,
        args: Optional[TrainingArguments] = None,
        train_dataset: Optional["Dataset"] = None,
        eval_dataset: Optional["Dataset"] = None,
        model_init: Optional[Callable[[], "SetFitModel"]] = None,
        metric: Union[str, Callable[["Dataset", "Dataset"], Dict[str, float]]] = "accuracy",
        column_mapping: Optional[Dict[str, str]] = None,
    ):
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.model_init = model_init
        self.metric = metric
        self.column_mapping = column_mapping

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
            missing_columns = self._REQUIRED_COLUMNS.difference(self.column_mapping.values())
            if missing_columns:
                raise ValueError(
                    f"The following columns are missing from the column mapping: {missing_columns}. Please provide a mapping for all required columns."
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
                **{col: f"feat_{col}" for col in dataset.column_names if col not in column_mapping},
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

    def apply_hyperparameters(self, params: Dict[str, Any], final_model: bool = False):
        """Applies a dictionary of hyperparameters to both the trainer and the model

        Args:
            params (`Dict[str, Any]`): The parameters, usually from `BestRun.hyperparameters`
            final_model (`bool`, *optional*, defaults to `False`): If `True`, replace the `model_init()` function with a fixed model based on the parameters.
        """

        if self.args is not None:
            self.args = self.args.update(params, ignore_extra=True)
        else:
            self.args = TrainingArguments.from_dict(params, ignore_extra=True)

        self.model = self.model_init(params)
        if final_model:
            self.model_init = None

    def _hp_search_setup(self, trial: Union["optuna.Trial", Dict[str, Any]]):
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

    def call_model_init(self, params: Optional[Dict[str, Any]] = None):
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
    ):
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

        args = args or self.args or TrainingArguments()

        set_seed(args.seed)  # Seed must be set before instantiating the model when using model_init.

        if trial:  # Trial and model initialization
            self._hp_search_setup(trial)  # sets trainer parameters and initializes model

        if self.train_dataset is None:
            raise ValueError(
                f"Training requires a `train_dataset` given to the `{self.__class__.__name__}` initialization."
            )

        self._validate_column_mapping(self.train_dataset)
        train_dataset = self.train_dataset
        if self.column_mapping is not None:
            logger.info("Applying column mapping to training dataset")
            train_dataset = self._apply_column_mapping(self.train_dataset, self.column_mapping)

        x_train: List[str] = train_dataset["text"]
        y_train: List[int] = train_dataset["label"]

        self.train_embeddings(x_train, y_train, args)
        self.train_classifier(x_train, y_train, args)

    def train_embeddings(
        self, x_train: List[str], y_train: Union[List[int], List[List[int]]], args: Optional[TrainingArguments] = None
    ):
        """
        Method to perform the embedding phase: finetuning the `SentenceTransformer` body.

        Args:
            x_train (`List[str]`): A list of training sentences.
            y_train (`Union[List[int], List[List[int]]]`): A list of labels corresponding to the training sentences.
            args (`TrainingArguments`, *optional*):
                Temporarily change the training arguments for this training call.
        """
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
            train_data_sampler = SentenceLabelDataset(train_examples, samples_per_label=args.samples_per_label)

            batch_size = min(args.embedding_batch_size, len(train_data_sampler))
            train_dataloader = DataLoader(train_data_sampler, batch_size=batch_size, drop_last=True)

            if args.loss is losses.BatchHardSoftMarginTripletLoss:
                train_loss = args.loss(
                    model=self.model.model_body,
                    distance_metric=args.distance_metric,
                )
            elif args.loss is SupConLoss:
                train_loss = args.loss(model=self.model.model_body)
            else:
                train_loss = args.loss(
                    model=self.model.model_body,
                    distance_metric=args.distance_metric,
                    margin=args.margin,
                )
        else:
            train_examples = []

            for _ in trange(args.num_iterations, desc="Generating Training Pairs", disable=not args.show_progress_bar):
                if self.model.multi_target_strategy is not None:
                    train_examples = sentence_pairs_generation_multilabel(
                        np.array(x_train), np.array(y_train), train_examples
                    )
                else:
                    train_examples = sentence_pairs_generation(np.array(x_train), np.array(y_train), train_examples)

            batch_size = args.embedding_batch_size
            train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
            train_loss = args.loss(self.model.model_body)

        total_train_steps = len(train_dataloader) * args.embedding_num_epochs
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_examples)}")
        logger.info(f"  Num epochs = {args.embedding_num_epochs}")
        logger.info(f"  Total optimization steps = {total_train_steps}")
        logger.info(f"  Total train batch size = {batch_size}")

        warmup_steps = math.ceil(total_train_steps * args.warmup_proportion)
        self.model.model_body.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=args.embedding_num_epochs,
            optimizer_params={"lr": args.body_embedding_learning_rate},
            warmup_steps=warmup_steps,
            show_progress_bar=args.show_progress_bar,
            use_amp=args.use_amp,
        )

    def train_classifier(
        self, x_train: List[str], y_train: Union[List[int], List[List[int]]], args: Optional[TrainingArguments] = None
    ):
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

    def evaluate(self):
        """
        Computes the metrics for a given classifier.

        Returns:
            `Dict[str, float]`: The evaluation metrics.
        """

        self._validate_column_mapping(self.eval_dataset)
        eval_dataset = self.eval_dataset

        if self.column_mapping is not None:
            logger.info("Applying column mapping to evaluation dataset")
            eval_dataset = self._apply_column_mapping(self.eval_dataset, self.column_mapping)

        x_test = eval_dataset["text"]
        y_test = eval_dataset["label"]

        logger.info("***** Running evaluation *****")
        y_pred = self.model.predict(x_test)

        if isinstance(self.metric, str):
            metric_config = "multilabel" if self.model.multi_target_strategy is not None else None
            metric_fn = evaluate.load(self.metric, config_name=metric_config)

            return metric_fn.compute(predictions=y_pred, references=y_test)

        elif callable(self.metric):
            return self.metric(y_pred, y_test)

        else:
            raise ValueError("metric must be a string or a callable")

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
                [`~trainer_utils.default_hp_space_optuna`].
            compute_objective (`Callable[[Dict[str, float]], float]`, *optional*):
                A function computing the objective to minimize or maximize from the metrics returned by the `evaluate`
                method. Will default to [`~trainer_utils.default_compute_objective`] which uses the sum of metrics.
            n_trials (`int`, *optional*, defaults to 100):
                The number of trial runs to test.
            direction (`str`, *optional*, defaults to `"maximize"`):
                Whether to optimize greater or lower objects. Can be `"minimize"` or `"maximize"`, you should pick
                `"minimize"` when optimizing the validation loss, `"maximize"` when optimizing one or several metrics.
            backend (`str` or [`~training_utils.HPSearchBackend`], *optional*):
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
                raise RuntimeError("optuna should be installed. To install optuna run `pip install optuna`. ")
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

    def push_to_hub(
        self,
        repo_path_or_name: Optional[str] = None,
        repo_url: Optional[str] = None,
        commit_message: Optional[str] = "Add SetFit model",
        organization: Optional[str] = None,
        private: Optional[bool] = None,
        api_endpoint: Optional[str] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
        git_user: Optional[str] = None,
        git_email: Optional[str] = None,
        config: Optional[dict] = None,
        skip_lfs_files: bool = False,
    ):
        return self.model.push_to_hub(
            repo_path_or_name,
            repo_url,
            commit_message,
            organization,
            private,
            api_endpoint,
            use_auth_token,
            git_user,
            git_email,
            config,
            skip_lfs_files,
        )


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
            column_mapping=column_mapping,
        )
