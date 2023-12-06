from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from datasets import Dataset
from transformers.trainer_callback import TrainerCallback

from setfit.span.modeling import AbsaModel, AspectModel, PolarityModel
from setfit.training_args import TrainingArguments

from .. import logging
from ..trainer import ColumnMappingMixin, Trainer


if TYPE_CHECKING:
    import optuna

logger = logging.get_logger(__name__)


class AbsaTrainer(ColumnMappingMixin):
    """Trainer to train a SetFit ABSA model.

    Args:
        model (`AbsaModel`):
            The AbsaModel model to train.
        args (`TrainingArguments`, *optional*):
            The training arguments to use. If `polarity_args` is not defined, then `args` is used for both
            the aspect and the polarity model.
        polarity_args (`TrainingArguments`, *optional*):
            The training arguments to use for the polarity model. If not defined, `args` is used for both
            the aspect and the polarity model.
        train_dataset (`Dataset`):
            The training dataset. The dataset must have "text", "span", "label" and "ordinal" columns.
        eval_dataset (`Dataset`, *optional*):
            The evaluation dataset. The dataset must have "text", "span", "label" and "ordinal" columns.
        metric (`str` or `Callable`, *optional*, defaults to `"accuracy"`):
            The metric to use for evaluation. If a string is provided, we treat it as the metric
            name and load it with default settings.
            If a callable is provided, it must take two arguments (`y_pred`, `y_test`).
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
            `{"text_column_name": "text", "span_column_name": "span", "label_column_name: "label", "ordinal_column_name": "ordinal"}`.
    """

    _REQUIRED_COLUMNS = {"text", "span", "label", "ordinal"}

    def __init__(
        self,
        model: AbsaModel,
        args: Optional[TrainingArguments] = None,
        polarity_args: Optional[TrainingArguments] = None,
        train_dataset: Optional["Dataset"] = None,
        eval_dataset: Optional["Dataset"] = None,
        metric: Union[str, Callable[["Dataset", "Dataset"], Dict[str, float]]] = "accuracy",
        metric_kwargs: Optional[Dict[str, Any]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        column_mapping: Optional[Dict[str, str]] = None,
    ) -> None:
        self.model = model
        self.aspect_extractor = model.aspect_extractor

        if train_dataset is not None and column_mapping:
            train_dataset = self._apply_column_mapping(train_dataset, column_mapping)
        aspect_train_dataset, polarity_train_dataset = self.preprocess_dataset(
            model.aspect_model, model.polarity_model, train_dataset
        )
        if eval_dataset is not None and column_mapping:
            eval_dataset = self._apply_column_mapping(eval_dataset, column_mapping)
        aspect_eval_dataset, polarity_eval_dataset = self.preprocess_dataset(
            model.aspect_model, model.polarity_model, eval_dataset
        )

        self.aspect_trainer = Trainer(
            model.aspect_model,
            args=args,
            train_dataset=aspect_train_dataset,
            eval_dataset=aspect_eval_dataset,
            metric=metric,
            metric_kwargs=metric_kwargs,
            callbacks=callbacks,
        )
        self.aspect_trainer._set_logs_mapper(
            {
                "eval_embedding_loss": "eval_aspect_embedding_loss",
                "embedding_loss": "aspect_embedding_loss",
            }
        )
        self.polarity_trainer = Trainer(
            model.polarity_model,
            args=polarity_args or args,
            train_dataset=polarity_train_dataset,
            eval_dataset=polarity_eval_dataset,
            metric=metric,
            metric_kwargs=metric_kwargs,
            callbacks=callbacks,
        )
        self.polarity_trainer._set_logs_mapper(
            {
                "eval_embedding_loss": "eval_polarity_embedding_loss",
                "embedding_loss": "polarity_embedding_loss",
            }
        )

    def preprocess_dataset(
        self, aspect_model: AspectModel, polarity_model: PolarityModel, dataset: Dataset
    ) -> Dataset:
        if dataset is None:
            return dataset, dataset

        # Group by "text"
        grouped_data = defaultdict(list)
        for sample in dataset:
            text = sample.pop("text")
            grouped_data[text].append(sample)

        def index_ordinal(text: str, target: str, ordinal: int) -> Tuple[int, int]:
            find_from = 0
            for _ in range(ordinal + 1):
                start_idx = text.index(target, find_from)
                find_from = start_idx + 1
            return start_idx, start_idx + len(target)

        def overlaps(aspect: slice, aspects: List[slice]) -> bool:
            for test_aspect in aspects:
                overlapping_indices = set(range(aspect.start, aspect.stop + 1)) & set(
                    range(test_aspect.start, test_aspect.stop + 1)
                )
                if overlapping_indices:
                    return True
            return False

        docs, aspects_list = self.aspect_extractor(grouped_data.keys())
        aspect_aspect_list = []
        aspect_labels = []
        polarity_aspect_list = []
        polarity_labels = []
        for doc, aspects, text in zip(docs, aspects_list, grouped_data):
            # Collect all of the gold aspects
            gold_aspects = []
            gold_polarity_labels = []
            for annotation in grouped_data[text]:
                try:
                    start, end = index_ordinal(text, annotation["span"], annotation["ordinal"])
                except ValueError:
                    logger.info(
                        f"The ordinal of {annotation['ordinal']} for span {annotation['span']!r} in {text!r} is too high. "
                        "Skipping this sample."
                    )
                    continue

                gold_aspect_span = doc.char_span(start, end)
                if gold_aspect_span is None:
                    continue
                gold_aspects.append(slice(gold_aspect_span.start, gold_aspect_span.end))
                gold_polarity_labels.append(annotation["label"])

            # The Aspect model uses all gold aspects as "True", and all non-overlapping predicted
            # aspects as "False"
            aspect_labels.extend([True] * len(gold_aspects))
            aspect_aspect_list.append(gold_aspects[:])
            for aspect in aspects:
                if not overlaps(aspect, gold_aspects):
                    aspect_labels.append(False)
                    aspect_aspect_list[-1].append(aspect)

            # The Polarity model uses only the gold aspects and labels
            polarity_labels.extend(gold_polarity_labels)
            polarity_aspect_list.append(gold_aspects)

        aspect_texts = list(aspect_model.prepend_aspects(docs, aspect_aspect_list))
        polarity_texts = list(polarity_model.prepend_aspects(docs, polarity_aspect_list))
        return Dataset.from_dict({"text": aspect_texts, "label": aspect_labels}), Dataset.from_dict(
            {"text": polarity_texts, "label": polarity_labels}
        )

    def train(
        self,
        args: Optional[TrainingArguments] = None,
        polarity_args: Optional[TrainingArguments] = None,
        trial: Optional[Union["optuna.Trial", Dict[str, Any]]] = None,
        **kwargs,
    ) -> None:
        """
        Main training entry point.

        Args:
            args (`TrainingArguments`, *optional*):
                Temporarily change the aspect training arguments for this training call.
            polarity_args (`TrainingArguments`, *optional*):
                Temporarily change the polarity training arguments for this training call.
            trial (`optuna.Trial` or `Dict[str, Any]`, *optional*):
                The trial run or the hyperparameter dictionary for hyperparameter search.
        """
        self.train_aspect(args=args, trial=trial, **kwargs)
        self.train_polarity(args=polarity_args, trial=trial, **kwargs)

    def train_aspect(
        self,
        args: Optional[TrainingArguments] = None,
        trial: Optional[Union["optuna.Trial", Dict[str, Any]]] = None,
        **kwargs,
    ) -> None:
        """
        Train the aspect model only.

        Args:
            args (`TrainingArguments`, *optional*):
                Temporarily change the aspect training arguments for this training call.
            trial (`optuna.Trial` or `Dict[str, Any]`, *optional*):
                The trial run or the hyperparameter dictionary for hyperparameter search.
        """
        self.aspect_trainer.train(args=args, trial=trial, **kwargs)

    def train_polarity(
        self,
        args: Optional[TrainingArguments] = None,
        trial: Optional[Union["optuna.Trial", Dict[str, Any]]] = None,
        **kwargs,
    ) -> None:
        """
        Train the polarity model only.

        Args:
            args (`TrainingArguments`, *optional*):
                Temporarily change the aspect training arguments for this training call.
            trial (`optuna.Trial` or `Dict[str, Any]`, *optional*):
                The trial run or the hyperparameter dictionary for hyperparameter search.
        """
        self.polarity_trainer.train(args=args, trial=trial, **kwargs)

    def add_callback(self, callback: Union[type, TrainerCallback]) -> None:
        """
        Add a callback to the current list of [`~transformers.TrainerCallback`].

        Args:
            callback (`type` or [`~transformers.TrainerCallback`]):
                A [`~transformers.TrainerCallback`] class or an instance of a [`~transformers.TrainerCallback`]. In the
                first case, will instantiate a member of that class.
        """
        self.aspect_trainer.add_callback(callback)
        self.polarity_trainer.add_callback(callback)

    def pop_callback(self, callback: Union[type, TrainerCallback]) -> Tuple[TrainerCallback, TrainerCallback]:
        """
        Remove a callback from the current list of [`~transformers.TrainerCallback`] and returns it.

        If the callback is not found, returns `None` (and no error is raised).

        Args:
            callback (`type` or [`~transformers.TrainerCallback`]):
                A [`~transformers.TrainerCallback`] class or an instance of a [`~transformers.TrainerCallback`]. In the
                first case, will pop the first member of that class found in the list of callbacks.

        Returns:
            `Tuple[`[`~transformers.TrainerCallback`], [`~transformers.TrainerCallback`]`]`: The callbacks removed from the
                aspect and polarity trainers, if found.
        """
        return self.aspect_trainer.pop_callback(callback), self.polarity_trainer.pop_callback(callback)

    def remove_callback(self, callback: Union[type, TrainerCallback]) -> None:
        """
        Remove a callback from the current list of [`~transformers.TrainerCallback`].

        Args:
            callback (`type` or [`~transformers.TrainerCallback`]):
                A [`~transformers.TrainerCallback`] class or an instance of a [`~transformers.TrainerCallback`]. In the
                first case, will remove the first member of that class found in the list of callbacks.
        """
        self.aspect_trainer.remove_callback(callback)
        self.polarity_trainer.remove_callback(callback)

    def push_to_hub(self, repo_id: str, polarity_repo_id: Optional[str] = None, **kwargs) -> None:
        """Upload model checkpoint to the Hub using `huggingface_hub`.

        See the full list of parameters for your `huggingface_hub` version in the\
        [huggingface_hub documentation](https://huggingface.co/docs/huggingface_hub/package_reference/mixins#huggingface_hub.ModelHubMixin.push_to_hub).

        Args:
            repo_id (`str`):
                The full repository ID to push to, e.g. `"tomaarsen/setfit-aspect"`.
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
        """
        return self.model.push_to_hub(repo_id=repo_id, polarity_repo_id=polarity_repo_id, **kwargs)

    def evaluate(self, dataset: Optional[Dataset] = None) -> Dict[str, Dict[str, float]]:
        """
        Computes the metrics for a given classifier.

        Args:
            dataset (`Dataset`, *optional*):
                The dataset to compute the metrics on. If not provided, will use the evaluation dataset passed via
                the `eval_dataset` argument at `Trainer` initialization.

        Returns:
            `Dict[str, Dict[str, float]]`: The evaluation metrics.
        """
        aspect_eval_dataset = polarity_eval_dataset = None
        if dataset:
            aspect_eval_dataset, polarity_eval_dataset = self.preprocess_dataset(
                self.model.aspect_model, self.model.polarity_model, dataset
            )
        return {
            "aspect": self.aspect_trainer.evaluate(aspect_eval_dataset),
            "polarity": self.polarity_trainer.evaluate(polarity_eval_dataset),
        }
