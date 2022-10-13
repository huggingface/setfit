import math
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Union

import evaluate
import numpy as np
from sentence_transformers import InputExample, losses
from sentence_transformers.datasets import SentenceLabelDataset
from sentence_transformers.losses.BatchHardTripletLoss import BatchHardTripletLossDistanceFunction
from torch.utils.data import DataLoader
from transformers.trainer_utils import HPSearchBackend, default_compute_objective, number_of_arguments, set_seed

from . import logging
from .integrations import default_hp_search_backend, is_optuna_available, run_hp_search_optuna
from .modeling import SupConLoss, sentence_pairs_generation, sentence_pairs_generation_multilabel
from .utils import BestRun, default_hp_space_optuna


if TYPE_CHECKING:
    import optuna
    from datasets import Dataset

    from .modeling import SetFitModel

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


class SetFitTrainer:
    """Trainer to train a SetFit model.

    Args:
        model (`SetFitModel`, *optional*):
            The model to train. If not provided, a `model_init` must be passed.
        train_dataset (`Dataset`):
            The training dataset.
        eval_dataset (`Dataset`, *optional*):
            The evaluation dataset.
        model_init (`Callable[[], SetFitModel]`, *optional*):
            A function that instantiates the model to be used. If provided, each call to [`~SetFitTrainer.train`] will start
            from a new instance of the model as given by this function when a `trial` is passed.
        metric (`str`, *optional*, defaults to `"accuracy"`):
            The metric to use for evaluation.
        loss_class (`nn.Module`, *optional*, defaults to `CosineSimilarityLoss`):
            The loss function to use for contrastive training.
        num_iterations (`int`, *optional*, defaults to `20`):
            The number of iterations to generate sentence pairs for.
        num_epochs (`int`, *optional*, defaults to `1`):
            The number of epochs to train the Sentence Transformer body for.
        learning_rate (`float`, *optional*, defaults to `2e-5`):
            The learning rate to use for contrastive training.
        batch_size (`int`, *optional*, defaults to `16`):
            The batch size to use for contrastive training.
        seed (`int`, *optional*, defaults to 42):
            Random seed that will be set at the beginning of training. To ensure reproducibility across runs, use the
            [`~SetTrainer.model_init`] function to instantiate the model if it has some randomly initialized parameters.
        column_mapping (`Dict[str, str]`, *optional*):
            A mapping from the column names in the dataset to the column names expected by the model. The expected format is a dictionary with the following format: {"text_column_name": "text", "label_column_name: "label"}.
    """

    def __init__(
        self,
        model: "SetFitModel" = None,
        train_dataset: "Dataset" = None,
        eval_dataset: "Dataset" = None,
        model_init: Callable[[], "SetFitModel"] = None,
        metric: str = "accuracy",
        loss_class=losses.CosineSimilarityLoss,
        num_iterations: int = 20,
        num_epochs: int = 1,
        learning_rate: float = 2e-5,
        batch_size: int = 16,
        seed: int = 42,
        column_mapping: Dict[str, str] = None,
    ):

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.metric = metric
        self.loss_class = loss_class
        self.num_iterations = num_iterations
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.seed = seed
        self.column_mapping = column_mapping

        if model is None:
            if model_init is not None:
                self.model_init = model_init
                model = self.call_model_init()
            else:
                raise RuntimeError("`SetFitTrainer` requires either a `model` or `model_init` argument")
        else:
            if model_init is not None:
                raise RuntimeError("`SetFitTrainer` requires either a `model` or `model_init` argument, but not both")

            self.model_init = model_init

        self.model = model
        self.hp_search_backend = None

    def _validate_column_mapping(self, dataset: "Dataset") -> None:
        """
        Validates the provided column mapping against the dataset.
        """
        required_columns = {"text", "label"}
        column_names = set(dataset.column_names)
        if self.column_mapping is None and not required_columns.issubset(column_names):
            raise ValueError(
                f"A column mapping must be provided when the dataset does not contain the following columns: {required_columns}"
            )
        if self.column_mapping is not None:
            missing_columns = required_columns.difference(self.column_mapping.values())
            if missing_columns:
                raise ValueError(
                    f"The following columns are missing from the column mapping: {missing_columns}. Please provide a mapping for all required columns."
                )
            if not set(self.column_mapping.keys()).issubset(column_names):
                raise ValueError(
                    f"The following columns are missing from the dataset: {set(self.column_mapping.keys()).difference(column_names)}. Please provide a mapping for all required columns."
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
        for key, value in params.items():
            if hasattr(self, key):
                old_attr = getattr(self, key, None)
                # Casting value to the proper type
                if old_attr is not None:
                    value = type(old_attr)(value)
                setattr(self, key, value)
            elif number_of_arguments(self.model_init) == 0:  # we do not warn if model_init could be using it
                logger.warning(
                    f"Trying to set {key} in the hyperparameter search but there is no corresponding field in "
                    "`SetFitTrainer`, and `model_init` does not take any arguments."
                )

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

    def call_model_init(self, params: Dict[str, Any] = None):
        model_init_argcount = number_of_arguments(self.model_init)
        if model_init_argcount == 0:
            model = self.model_init()
        elif model_init_argcount == 1:
            model = self.model_init(params)
        else:
            raise RuntimeError("model_init should have 0 or 1 argument.")

        if model is None:
            raise RuntimeError("model_init should not return None.")

        return model

    def train(self, trial: Union["optuna.Trial", Dict[str, Any]] = None):
        """
        Main training entry point.

        Args:
            trial (`optuna.Trial` or `Dict[str, Any]`, *optional*):
                The trial run or the hyperparameter dictionary for hyperparameter search.
        """
        if trial:  # Trial and model initialization
            set_seed(self.seed)  # Seed must be set before instantiating the model when using model_init.
            self._hp_search_setup(trial)  # sets trainer parameters and initializes model

        if self.train_dataset is None:
            raise ValueError("SetFitTrainer: training requires a train_dataset.")

        self._validate_column_mapping(self.train_dataset)
        train_dataset = self.train_dataset
        if self.column_mapping is not None:
            logger.info("Applying column mapping to training dataset")
            train_dataset = self._apply_column_mapping(self.train_dataset, self.column_mapping)
        x_train = train_dataset["text"]
        y_train = train_dataset["label"]

        if self.loss_class is None:
            return

        # sentence-transformers adaptation
        batch_size = self.batch_size
        if self.loss_class in [
            losses.BatchAllTripletLoss,
            losses.BatchHardTripletLoss,
            losses.BatchSemiHardTripletLoss,
            losses.BatchHardSoftMarginTripletLoss,
            SupConLoss,
        ]:
            train_examples = [InputExample(texts=[text], label=label) for text, label in zip(x_train, y_train)]
            train_data_sampler = SentenceLabelDataset(train_examples)

            batch_size = min(self.batch_size, len(train_data_sampler))
            train_dataloader = DataLoader(train_data_sampler, batch_size=batch_size, drop_last=True)

            if self.loss_class is losses.BatchHardSoftMarginTripletLoss:
                train_loss = self.loss_class(
                    model=self.model,
                    distance_metric=BatchHardTripletLossDistanceFunction.cosine_distance,
                )
            elif self.loss_class is SupConLoss:
                train_loss = self.loss_class(model=self.model)
            else:
                train_loss = self.loss_class(
                    model=self.model,
                    distance_metric=BatchHardTripletLossDistanceFunction.cosine_distance,
                    margin=0.25,
                )

            train_steps = len(train_dataloader) * self.num_epochs
        else:
            train_examples = []

            for _ in range(self.num_iterations):
                if self.model.multi_target_strategy is not None:
                    train_examples = sentence_pairs_generation_multilabel(
                        np.array(x_train), np.array(y_train), train_examples
                    )
                else:
                    train_examples = sentence_pairs_generation(np.array(x_train), np.array(y_train), train_examples)

            train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=self.batch_size)
            train_loss = self.loss_class(self.model.model_body)
            train_steps = len(train_dataloader)

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_examples)}")
        logger.info(f"  Num epochs = {self.num_epochs}")
        logger.info(f"  Total optimization steps = {train_steps}")
        logger.info(f"  Total train batch size = {batch_size}")

        warmup_steps = math.ceil(train_steps * 0.1)
        self.model.model_body.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=self.num_epochs,
            steps_per_epoch=train_steps,
            optimizer_params={"lr": self.learning_rate},
            warmup_steps=warmup_steps,
            show_progress_bar=True,
        )

        # Train the final classifier
        self.model.fit(x_train, y_train)

    def evaluate(self):
        """Computes the metrics for a given classifier."""
        self._validate_column_mapping(self.eval_dataset)
        eval_dataset = self.eval_dataset
        if self.column_mapping is not None:
            logger.info("Applying column mapping to evaluation dataset")
            eval_dataset = self._apply_column_mapping(self.eval_dataset, self.column_mapping)
        metric_config = "multilabel" if self.model.multi_target_strategy is not None else None
        metric_fn = evaluate.load(self.metric, config_name=metric_config)
        x_test = eval_dataset["text"]
        y_test = eval_dataset["label"]

        logger.info("***** Running evaluation *****")
        y_pred = self.model.predict(x_test)

        return metric_fn.compute(predictions=y_pred, references=y_test)

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

        To use this method, you need to have provided a `model_init` when initializing your [`SetFitTrainer`]: we need to
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
                raise RuntimeError("optuna should be installed. " "To install optuna run `pip install optuna`. ")
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
        use_auth_token: Union[bool, str] = None,
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
