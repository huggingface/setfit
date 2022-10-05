import math
from typing import TYPE_CHECKING, Dict, Optional, Union

import evaluate
import numpy as np
from sentence_transformers import InputExample, losses
from sentence_transformers.datasets import SentenceLabelDataset
from sentence_transformers.losses.BatchHardTripletLoss import BatchHardTripletLossDistanceFunction
from torch.utils.data import DataLoader

from . import logging
from .modeling import SupConLoss, sentence_pairs_generation


if TYPE_CHECKING:
    from datasets import Dataset

    from .modeling import SetFitModel

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


class SetFitTrainer:
    """Trainer to train a SetFit model.

    Args:
        model (`SetFitModel`):
            The model to train.
        train_dataset (`Dataset`):
            The training dataset.
        eval_dataset (`Dataset`, *optional*):
            The evaluation dataset.
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
        column_mapping (`Dict[str, str]`, *optional*):
            A mapping from the column names in the dataset to the column names expected by the model. The expected format is a dictionary with the following format: {"text_column_name": "text", "label_column_name: "label"}.
    """

    def __init__(
        self,
        model: "SetFitModel",
        train_dataset: "Dataset",
        eval_dataset: "Dataset" = None,
        metric: str = "accuracy",
        loss_class=losses.CosineSimilarityLoss,
        num_iterations: int = 20,
        num_epochs: int = 1,
        learning_rate: float = 2e-5,
        batch_size: int = 16,
        column_mapping: Dict[str, str] = None,
    ):

        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.metric = metric
        self.loss_class = loss_class
        self.num_iterations = num_iterations
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.column_mapping = column_mapping

    def _validate_column_mapping(self, dataset: "Dataset") -> None:
        """
        Validates the provided column mapping against the dataset.
        """
        required_columns = set(["text", "label"])
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

    def train(self):
        self._validate_column_mapping(self.train_dataset)
        if self.column_mapping is not None:
            logger.info("Applying column mapping to training dataset")
            self.train_dataset = self._apply_column_mapping(self.train_dataset, self.column_mapping)
        x_train = self.train_dataset["text"]
        y_train = self.train_dataset["label"]

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
        if self.column_mapping is not None:
            logger.info("Applying column mapping to evaluation dataset")
            self.eval_dataset = self._apply_column_mapping(self.eval_dataset, self.column_mapping)
        metric_fn = evaluate.load(self.metric)
        x_test = self.eval_dataset["text"]
        y_test = self.eval_dataset["label"]

        logger.info("***** Running evaluation *****")
        y_pred = self.model.predict(x_test)
        return metric_fn.compute(predictions=y_pred, references=y_test)

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
