import json
import os
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Set, Tuple, Union

import joblib
import numpy as np
import requests
import torch
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
from huggingface_hub.utils import validate_hf_hub_args
from packaging.version import Version, parse
from sentence_transformers import SentenceTransformer
from sentence_transformers import __version__ as sentence_transformers_version
from sentence_transformers import models
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange
from transformers.utils import copy_func

from . import logging
from .data import SetFitDataset
from .model_card import SetFitModelCardData, generate_model_card
from .utils import set_docstring


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

MODEL_HEAD_NAME = "model_head.pkl"
CONFIG_NAME = "config_setfit.json"


class SetFitHead(models.Dense):
    """
    A SetFit head that supports multi-class classification for end-to-end training.
    Binary classification is treated as 2-class classification.

    To be compatible with Sentence Transformers, we inherit `Dense` from:
    https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/models/Dense.py

    Args:
        in_features (`int`, *optional*):
            The embedding dimension from the output of the SetFit body. If `None`, defaults to `LazyLinear`.
        out_features (`int`, defaults to `2`):
            The number of targets. If set `out_features` to 1 for binary classification, it will be changed to 2 as 2-class classification.
        temperature (`float`, defaults to `1.0`):
            A logits' scaling factor. Higher values make the model less confident and lower values make
            it more confident.
        eps (`float`, defaults to `1e-5`):
            A value for numerical stability when scaling logits.
        bias (`bool`, *optional*, defaults to `True`):
            Whether to add bias to the head.
        device (`torch.device`, str, *optional*):
            The device the model will be sent to. If `None`, will check whether GPU is available.
        multitarget (`bool`, defaults to `False`):
            Enable multi-target classification by making `out_features` binary predictions instead
            of a single multinomial prediction.
    """

    def __init__(
        self,
        in_features: Optional[int] = None,
        out_features: int = 2,
        temperature: float = 1.0,
        eps: float = 1e-5,
        bias: bool = True,
        device: Optional[Union[torch.device, str]] = None,
        multitarget: bool = False,
    ) -> None:
        super(models.Dense, self).__init__()  # init on models.Dense's parent: nn.Module

        if out_features == 1:
            logger.warning(
                "Change `out_features` from 1 to 2 since we use `CrossEntropyLoss` for binary classification."
            )
            out_features = 2

        if in_features is not None:
            self.linear = nn.Linear(in_features, out_features, bias=bias)
        else:
            self.linear = nn.LazyLinear(out_features, bias=bias)

        self.in_features = in_features
        self.out_features = out_features
        self.temperature = temperature
        self.eps = eps
        self.bias = bias
        self._device = device or "cuda" if torch.cuda.is_available() else "cpu"
        self.multitarget = multitarget

        self.to(self._device)
        self.apply(self._init_weight)

    def forward(
        self,
        features: Union[Dict[str, torch.Tensor], torch.Tensor],
        temperature: Optional[float] = None,
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor]]:
        """
        SetFitHead can accept embeddings in:
        1. Output format (`dict`) from Sentence-Transformers.
        2. Pure `torch.Tensor`.

        Args:
            features (`Dict[str, torch.Tensor]` or `torch.Tensor):
                The embeddings from the encoder. If using `dict` format,
                make sure to store embeddings under the key: 'sentence_embedding'
                and the outputs will be under the key: 'prediction'.
            temperature (`float`, *optional*):
                A logits' scaling factor. Higher values make the model less
                confident and lower values make it more confident.
                Will override the temperature given during initialization.
        Returns:
            [`Dict[str, torch.Tensor]` or `Tuple[torch.Tensor]`]
        """
        temperature = temperature or self.temperature
        is_features_dict = False  # whether `features` is dict or not
        if isinstance(features, dict):
            assert "sentence_embedding" in features
            is_features_dict = True
        x = features["sentence_embedding"] if is_features_dict else features
        logits = self.linear(x)
        logits = logits / (temperature + self.eps)
        if self.multitarget:  # multiple targets per item
            probs = torch.sigmoid(logits)
        else:  # one target per item
            probs = nn.functional.softmax(logits, dim=-1)
        if is_features_dict:
            features.update(
                {
                    "logits": logits,
                    "probs": probs,
                }
            )
            return features

        return logits, probs

    def predict_proba(self, x_test: torch.Tensor) -> torch.Tensor:
        self.eval()

        return self(x_test)[1]

    def predict(self, x_test: torch.Tensor) -> torch.Tensor:
        probs = self.predict_proba(x_test)

        if self.multitarget:
            return torch.where(probs >= 0.5, 1, 0)
        return torch.argmax(probs, dim=-1)

    def get_loss_fn(self) -> nn.Module:
        if self.multitarget:  # if sigmoid output
            return torch.nn.BCEWithLogitsLoss()
        return torch.nn.CrossEntropyLoss()

    @property
    def device(self) -> torch.device:
        """
        `torch.device`: The device on which the model is placed.

        Reference from: https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/SentenceTransformer.py#L869
        """
        return next(self.parameters()).device

    def get_config_dict(self) -> Dict[str, Optional[Union[int, float, bool]]]:
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "temperature": self.temperature,
            "bias": self.bias,
            "device": self.device.type,  # store the string of the device, instead of `torch.device`
        }

    @staticmethod
    def _init_weight(module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 1e-2)

    def __repr__(self) -> str:
        return "SetFitHead({})".format(self.get_config_dict())


@dataclass
class SetFitModel(PyTorchModelHubMixin):
    """A SetFit model with integration to the [Hugging Face Hub](https://huggingface.co).

    Example::

        >>> from setfit import SetFitModel
        >>> model = SetFitModel.from_pretrained("tomaarsen/setfit-bge-small-v1.5-sst2-8-shot")
        >>> model.predict([
        ...     "It's a charming and often affecting journey.",
        ...     "It's slow -- very, very slow.",
        ...     "A sometimes tedious film.",
        ... ])
        ['positive', 'negative', 'negative']
    """

    model_body: Optional[SentenceTransformer] = None
    model_head: Optional[Union[SetFitHead, LogisticRegression]] = None
    multi_target_strategy: Optional[str] = None
    normalize_embeddings: bool = False
    labels: Optional[List[str]] = None
    model_card_data: Optional[SetFitModelCardData] = field(default_factory=SetFitModelCardData)
    sentence_transformers_kwargs: Dict = field(default_factory=dict, repr=False)

    attributes_to_save: Set[str] = field(
        init=False, repr=False, default_factory=lambda: {"normalize_embeddings", "labels"}
    )

    def __post_init__(self):
        self.model_card_data.register_model(self)

    @property
    def has_differentiable_head(self) -> bool:
        # if False, sklearn is assumed to be used instead
        return isinstance(self.model_head, nn.Module)

    @property
    def id2label(self) -> Dict[int, str]:
        """Return a mapping from integer IDs to string labels."""
        if self.labels is None:
            return {}
        return dict(enumerate(self.labels))

    @property
    def label2id(self) -> Dict[str, int]:
        """Return a mapping from string labels to integer IDs."""
        if self.labels is None:
            return {}
        return {label: idx for idx, label in enumerate(self.labels)}

    def fit(
        self,
        x_train: List[str],
        y_train: Union[List[int], List[List[int]]],
        num_epochs: int,
        batch_size: Optional[int] = None,
        body_learning_rate: Optional[float] = None,
        head_learning_rate: Optional[float] = None,
        end_to_end: bool = False,
        l2_weight: Optional[float] = None,
        max_length: Optional[int] = None,
        show_progress_bar: bool = True,
    ) -> None:
        """Train the classifier head, only used if a differentiable PyTorch head is used.

        Args:
            x_train (`List[str]`): A list of training sentences.
            y_train (`Union[List[int], List[List[int]]]`): A list of labels corresponding to the training sentences.
            num_epochs (`int`): The number of epochs to train for.
            batch_size (`int`, *optional*): The batch size to use.
            body_learning_rate (`float`, *optional*): The learning rate for the `SentenceTransformer` body
                in the `AdamW` optimizer. Disregarded if `end_to_end=False`.
            head_learning_rate (`float`, *optional*): The learning rate for the differentiable torch head
                in the `AdamW` optimizer.
            end_to_end (`bool`, defaults to `False`): If True, train the entire model end-to-end.
                Otherwise, freeze the `SentenceTransformer` body and only train the head.
            l2_weight (`float`, *optional*): The l2 weight for both the model body and head
                in the `AdamW` optimizer.
            max_length (`int`, *optional*): The maximum token length a tokenizer can generate. If not provided,
                the maximum length for the `SentenceTransformer` body is used.
            show_progress_bar (`bool`, defaults to `True`): Whether to display a progress bar for the training
                epochs and iterations.
        """
        if self.has_differentiable_head:  # train with pyTorch
            self.model_body.train()
            self.model_head.train()
            if not end_to_end:
                self.freeze("body")

            dataloader = self._prepare_dataloader(x_train, y_train, batch_size, max_length)
            criterion = self.model_head.get_loss_fn()
            optimizer = self._prepare_optimizer(head_learning_rate, body_learning_rate, l2_weight)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
            for epoch_idx in trange(num_epochs, desc="Epoch", disable=not show_progress_bar):
                for batch in tqdm(dataloader, desc="Iteration", disable=not show_progress_bar, leave=False):
                    features, labels = batch
                    optimizer.zero_grad()

                    # to model's device
                    features = {k: v.to(self.device) for k, v in features.items()}
                    labels = labels.to(self.device)

                    outputs = self.model_body(features)
                    if self.normalize_embeddings:
                        outputs["sentence_embedding"] = nn.functional.normalize(
                            outputs["sentence_embedding"], p=2, dim=1
                        )
                    outputs = self.model_head(outputs)
                    logits = outputs["logits"]

                    loss: torch.Tensor = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()

                scheduler.step()

            if not end_to_end:
                self.unfreeze("body")
        else:  # train with sklearn
            embeddings = self.model_body.encode(x_train, normalize_embeddings=self.normalize_embeddings)
            self.model_head.fit(embeddings, y_train)
            if self.labels is None and self.multi_target_strategy is None:
                # Try to set the labels based on the head classes, if they exist
                # This can fail in various ways, so we catch all exceptions
                try:
                    classes = self.model_head.classes_
                    if classes.dtype.char == "U":
                        self.labels = classes.tolist()
                except Exception:
                    pass

    def _prepare_dataloader(
        self,
        x_train: List[str],
        y_train: Union[List[int], List[List[int]]],
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        shuffle: bool = True,
    ) -> DataLoader:
        max_acceptable_length = self.model_body.get_max_seq_length()
        if max_length is None:
            max_length = max_acceptable_length
            logger.warning(
                f"The `max_length` is `None`. Using the maximum acceptable length according to the current model body: {max_length}."
            )

        if max_length > max_acceptable_length:
            logger.warning(
                (
                    f"The specified `max_length`: {max_length} is greater than the maximum length of the current model body: {max_acceptable_length}. "
                    f"Using {max_acceptable_length} instead."
                )
            )
            max_length = max_acceptable_length

        dataset = SetFitDataset(
            x_train,
            y_train,
            tokenizer=self.model_body.tokenizer,
            max_length=max_length,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=shuffle,
            pin_memory=True,
        )

        return dataloader

    def _prepare_optimizer(
        self,
        head_learning_rate: float,
        body_learning_rate: Optional[float],
        l2_weight: float,
    ) -> torch.optim.Optimizer:
        body_learning_rate = body_learning_rate or head_learning_rate
        l2_weight = l2_weight or 1e-2
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": self.model_body.parameters(),
                    "lr": body_learning_rate,
                    "weight_decay": l2_weight,
                },
                {"params": self.model_head.parameters(), "lr": head_learning_rate, "weight_decay": l2_weight},
            ],
        )

        return optimizer

    def freeze(self, component: Optional[Literal["body", "head"]] = None) -> None:
        """Freeze the model body and/or the head, preventing further training on that component until unfrozen.

        Args:
            component (`Literal["body", "head"]`, *optional*): Either "body" or "head" to freeze that component.
                If no component is provided, freeze both. Defaults to None.
        """
        if component is None or component == "body":
            self._freeze_or_not(self.model_body, to_freeze=True)

        if (component is None or component == "head") and self.has_differentiable_head:
            self._freeze_or_not(self.model_head, to_freeze=True)

    def unfreeze(
        self, component: Optional[Literal["body", "head"]] = None, keep_body_frozen: Optional[bool] = None
    ) -> None:
        """Unfreeze the model body and/or the head, allowing further training on that component.

        Args:
            component (`Literal["body", "head"]`, *optional*): Either "body" or "head" to unfreeze that component.
                If no component is provided, unfreeze both. Defaults to None.
            keep_body_frozen (`bool`, *optional*): Deprecated argument, use `component` instead.
        """
        if keep_body_frozen is not None:
            warnings.warn(
                "`keep_body_frozen` is deprecated and will be removed in v2.0.0 of SetFit. "
                'Please either pass "head", "body" or no arguments to unfreeze both.',
                DeprecationWarning,
                stacklevel=2,
            )
            # If the body must stay frozen, only unfreeze the head. Eventually, this entire if-branch
            # can be removed.
            if keep_body_frozen and not component:
                component = "head"

        if component is None or component == "body":
            self._freeze_or_not(self.model_body, to_freeze=False)

        if (component is None or component == "head") and self.has_differentiable_head:
            self._freeze_or_not(self.model_head, to_freeze=False)

    def _freeze_or_not(self, model: nn.Module, to_freeze: bool) -> None:
        """Set `requires_grad=not to_freeze` for all parameters in `model`"""
        for param in model.parameters():
            param.requires_grad = not to_freeze

    def encode(
        self, inputs: List[str], batch_size: int = 32, show_progress_bar: Optional[bool] = None
    ) -> Union[torch.Tensor, np.ndarray]:
        """Convert input sentences to embeddings using the `SentenceTransformer` body.

        Args:
            inputs (`List[str]`): The input sentences to embed.
            batch_size (`int`, defaults to `32`): The batch size to use in encoding the sentences to embeddings.
                Higher often means faster processing but higher memory usage.
            show_progress_bar (`Optional[bool]`, defaults to `None`): Whether to show a progress bar while encoding.

        Returns:
            Union[torch.Tensor, np.ndarray]: A matrix with shape [INPUT_LENGTH, EMBEDDING_SIZE], as a
            torch Tensor if this model has a differentiable Torch head, or otherwise as a numpy array.
        """
        return self.model_body.encode(
            inputs,
            batch_size=batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_tensor=self.has_differentiable_head,
            show_progress_bar=show_progress_bar,
        )

    def _output_type_conversion(
        self, outputs: Union[torch.Tensor, np.ndarray], as_numpy: bool = False
    ) -> Union[torch.Tensor, np.ndarray]:
        """Return `outputs` in the desired type:
        * Numpy array if no differentiable head is used.
        * Torch tensor if a differentiable head is used.

        Note:
            If the model is trained with string labels, which is only possible with a non-differentiable head,
            then we cannot output using torch Tensors, but only using a numpy array.

        Returns:
            Union[torch.Tensor, "ndarray"]: The input, correctly converted to the desired type.
        """
        if as_numpy and self.has_differentiable_head:
            outputs = outputs.detach().cpu().numpy()
        elif not as_numpy and not self.has_differentiable_head and outputs.dtype.char != "U":
            # Only output as tensor if the output isn't a string
            outputs = torch.from_numpy(outputs)
        return outputs

    def predict_proba(
        self,
        inputs: Union[str, List[str]],
        batch_size: int = 32,
        as_numpy: bool = False,
        show_progress_bar: Optional[bool] = None,
    ) -> Union[torch.Tensor, np.ndarray]:
        """Predict the probabilities of the various classes.

        Args:
            inputs (`Union[str, List[str]]`): The input sentences to predict class probabilities for.
            batch_size (`int`, defaults to `32`): The batch size to use in encoding the sentences to embeddings.
                Higher often means faster processing but higher memory usage.
            as_numpy (`bool`, defaults to `False`): Whether to output as numpy array instead.
            show_progress_bar (`Optional[bool]`, defaults to `None`): Whether to show a progress bar while encoding.

        Example::

            >>> model = SetFitModel.from_pretrained(...)
            >>> model.predict_proba(["What a boring display", "Exhilarating through and through", "I'm wowed!"])
            tensor([[0.9367, 0.0633],
                    [0.0627, 0.9373],
                    [0.0890, 0.9110]], dtype=torch.float64)
            >>> model.predict_proba("That was cool!")
            tensor([0.8421, 0.1579], dtype=torch.float64)

        Returns:
            `Union[torch.Tensor, np.ndarray]`: A matrix with shape [INPUT_LENGTH, NUM_CLASSES] denoting
            probabilities of predicting an input as a class. If the input is a string, then the output
            is a vector with shape [NUM_CLASSES,].
        """
        is_singular = isinstance(inputs, str)
        if is_singular:
            inputs = [inputs]
        embeddings = self.encode(inputs, batch_size=batch_size, show_progress_bar=show_progress_bar)
        probs = self.model_head.predict_proba(embeddings)
        if isinstance(probs, list):
            if self.has_differentiable_head:
                probs = torch.stack(probs, axis=1)
            else:
                probs = np.stack(probs, axis=1)
        outputs = self._output_type_conversion(probs, as_numpy=as_numpy)
        return outputs[0] if is_singular else outputs

    def predict(
        self,
        inputs: Union[str, List[str]],
        batch_size: int = 32,
        as_numpy: bool = False,
        use_labels: bool = True,
        show_progress_bar: Optional[bool] = None,
    ) -> Union[torch.Tensor, np.ndarray, List[str], int, str]:
        """Predict the various classes.

        Args:
            inputs (`Union[str, List[str]]`): The input sentence or sentences to predict classes for.
            batch_size (`int`, defaults to `32`): The batch size to use in encoding the sentences to embeddings.
                Higher often means faster processing but higher memory usage.
            as_numpy (`bool`, defaults to `False`): Whether to output as numpy array instead.
            use_labels (`bool`, defaults to `True`): Whether to try and return elements of `SetFitModel.labels`.
            show_progress_bar (`Optional[bool]`, defaults to `None`): Whether to show a progress bar while encoding.

        Example::

            >>> model = SetFitModel.from_pretrained(...)
            >>> model.predict(["What a boring display", "Exhilarating through and through", "I'm wowed!"])
            ["negative", "positive", "positive"]
            >>> model.predict("That was cool!")
            "positive"

        Returns:
            `Union[torch.Tensor, np.ndarray, List[str], int, str]`: A list of string labels with equal length to the
                inputs if `use_labels` is `True` and `SetFitModel.labels` has been defined. Otherwise a vector with
                equal length to the inputs, denoting to which class each input is predicted to belong. If the inputs
                is a single string, then the output is a single label as well.
        """
        is_singular = isinstance(inputs, str)
        if is_singular:
            inputs = [inputs]
        embeddings = self.encode(inputs, batch_size=batch_size, show_progress_bar=show_progress_bar)
        preds = self.model_head.predict(embeddings)
        # If labels are defined, we don't have multilabels & the output is not already strings, then we convert to string labels
        if (
            use_labels
            and self.labels
            and preds.ndim == 1
            and (self.has_differentiable_head or preds.dtype.char != "U")
        ):
            outputs = [self.labels[int(pred)] for pred in preds]
        else:
            outputs = self._output_type_conversion(preds, as_numpy=as_numpy)
        return outputs[0] if is_singular else outputs

    def __call__(
        self,
        inputs: Union[str, List[str]],
        batch_size: int = 32,
        as_numpy: bool = False,
        use_labels: bool = True,
        show_progress_bar: Optional[bool] = None,
    ) -> Union[torch.Tensor, np.ndarray, List[str], int, str]:
        """Predict the various classes.

        Args:
            inputs (`Union[str, List[str]]`): The input sentence or sentences to predict classes for.
            batch_size (`int`, defaults to `32`): The batch size to use in encoding the sentences to embeddings.
                Higher often means faster processing but higher memory usage.
            as_numpy (`bool`, defaults to `False`): Whether to output as numpy array instead.
            use_labels (`bool`, defaults to `True`): Whether to try and return elements of `SetFitModel.labels`.
            show_progress_bar (`Optional[bool]`, defaults to `None`): Whether to show a progress bar while encoding.

        Example::

            >>> model = SetFitModel.from_pretrained(...)
            >>> model(["What a boring display", "Exhilarating through and through", "I'm wowed!"])
            ["negative", "positive", "positive"]
            >>> model("That was cool!")
            "positive"

        Returns:
            `Union[torch.Tensor, np.ndarray, List[str], int, str]`: A list of string labels with equal length to the
                inputs if `use_labels` is `True` and `SetFitModel.labels` has been defined. Otherwise a vector with
                equal length to the inputs, denoting to which class each input is predicted to belong. If the inputs
                is a single string, then the output is a single label as well.
        """
        return self.predict(
            inputs,
            batch_size=batch_size,
            as_numpy=as_numpy,
            use_labels=use_labels,
            show_progress_bar=show_progress_bar,
        )

    @property
    def device(self) -> torch.device:
        """Get the Torch device that this model is on.

        Returns:
            torch.device: The device that the model is on.
        """
        # SentenceTransformers.device is reliable from 2.3.0 onwards
        if parse(sentence_transformers_version) >= Version("2.3.0"):
            return self.model_body.device
        return self.model_body._target_device

    def to(self, device: Union[str, torch.device]) -> "SetFitModel":
        """Move this SetFitModel to `device`, and then return `self`. This method does not copy.

        Args:
            device (Union[str, torch.device]): The identifier of the device to move the model to.

        Example::

            >>> model = SetFitModel.from_pretrained(...)
            >>> model.to("cpu")
            >>> model(["cats are cute", "dogs are loyal"])

        Returns:
            SetFitModel: Returns the original model, but now on the desired device.
        """
        # Note that we must also set _target_device with sentence-transformers <2.3.0,
        # or any SentenceTransformer.fit() call will reset the body location
        if parse(sentence_transformers_version) < Version("2.3.0"):
            self.model_body._target_device = device if isinstance(device, torch.device) else torch.device(device)
        self.model_body = self.model_body.to(device)

        if self.has_differentiable_head:
            self.model_head = self.model_head.to(device)

        return self

    def create_model_card(self, path: str, model_name: Optional[str] = "SetFit Model") -> None:
        """Creates and saves a model card for a SetFit model.

        Args:
            path (str): The path to save the model card to.
            model_name (str, *optional*): The name of the model. Defaults to `SetFit Model`.
        """
        if not os.path.exists(path):
            os.makedirs(path)

        # If the model_path is a folder that exists locally, i.e. when create_model_card is called
        # via push_to_hub, and the path is in a temporary folder, then we only take the last two
        # directories
        model_path = Path(model_name)
        if model_path.exists() and Path(tempfile.gettempdir()) in model_path.resolve().parents:
            self.model_card_data.model_id = "/".join(model_path.parts[-2:])

        with open(os.path.join(path, "README.md"), "w", encoding="utf-8") as f:
            f.write(self.generate_model_card())

    def generate_model_card(self) -> str:
        """Generate and return a model card string based on the model card data.

        Returns:
            str: The model card string.
        """
        return generate_model_card(self)

    def _save_pretrained(self, save_directory: Union[Path, str]) -> None:
        save_directory = str(save_directory)
        # Save the config
        config_path = os.path.join(save_directory, CONFIG_NAME)
        with open(config_path, "w") as f:
            json.dump(
                {
                    attr_name: getattr(self, attr_name)
                    for attr_name in self.attributes_to_save
                    if hasattr(self, attr_name)
                },
                f,
                indent=2,
            )
        # Save the body
        self.model_body.save(path=save_directory, create_model_card=False)
        # Save the README
        self.create_model_card(path=save_directory, model_name=save_directory)
        # Move the head to the CPU before saving
        if self.has_differentiable_head:
            self.model_head.to("cpu")
        # Save the classification head
        joblib.dump(self.model_head, str(Path(save_directory) / MODEL_HEAD_NAME))
        if self.has_differentiable_head:
            self.model_head.to(self.device)

    @classmethod
    @validate_hf_hub_args
    def _from_pretrained(
        cls,
        model_id: str,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        force_download: Optional[bool] = None,
        proxies: Optional[Dict] = None,
        resume_download: Optional[bool] = None,
        local_files_only: Optional[bool] = None,
        token: Optional[Union[bool, str]] = None,
        multi_target_strategy: Optional[str] = None,
        use_differentiable_head: bool = False,
        device: Optional[Union[torch.device, str]] = None,
        trust_remote_code: bool = False,
        **model_kwargs,
    ) -> "SetFitModel":
        sentence_transformers_kwargs = {
            "cache_folder": cache_dir,
            "use_auth_token": token,
            "device": device,
            "trust_remote_code": trust_remote_code,
        }
        if parse(sentence_transformers_version) >= Version("2.3.0"):
            sentence_transformers_kwargs = {
                "cache_folder": cache_dir,
                "token": token,
                "device": device,
                "trust_remote_code": trust_remote_code,
            }
        else:
            if trust_remote_code:
                raise ValueError(
                    "The `trust_remote_code` argument is only supported for `sentence-transformers` >= 2.3.0."
                )
            sentence_transformers_kwargs = {
                "cache_folder": cache_dir,
                "use_auth_token": token,
                "device": device,
            }
        model_body = SentenceTransformer(model_id, **sentence_transformers_kwargs)
        if parse(sentence_transformers_version) >= Version("2.3.0"):
            device = model_body.device
        else:
            device = model_body._target_device
        model_body.to(device)  # put `model_body` on the target device

        # Try to load a SetFit config file
        config_file: Optional[str] = None
        if os.path.isdir(model_id):
            if CONFIG_NAME in os.listdir(model_id):
                config_file = os.path.join(model_id, CONFIG_NAME)
        else:
            try:
                config_file = hf_hub_download(
                    repo_id=model_id,
                    filename=CONFIG_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except requests.exceptions.RequestException:
                pass

        model_kwargs = {key: value for key, value in model_kwargs.items() if value is not None}

        if config_file is not None:
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
            # Update model_kwargs + warnings
            for setting, value in config.items():
                if setting in model_kwargs:
                    if model_kwargs[setting] != value:
                        logger.warning(
                            f"Overriding {setting} in model configuration from {value} to {model_kwargs[setting]}."
                        )
                else:
                    model_kwargs[setting] = value

        # Try to load a model head file
        if os.path.isdir(model_id):
            if MODEL_HEAD_NAME in os.listdir(model_id):
                model_head_file = os.path.join(model_id, MODEL_HEAD_NAME)
            else:
                logger.info(
                    f"{MODEL_HEAD_NAME} not found in {Path(model_id).resolve()},"
                    " initialising classification head with random weights."
                    " You should TRAIN this model on a downstream task to use it for predictions and inference."
                )
                model_head_file = None
        else:
            try:
                model_head_file = hf_hub_download(
                    repo_id=model_id,
                    filename=MODEL_HEAD_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except requests.exceptions.RequestException:
                logger.info(
                    f"{MODEL_HEAD_NAME} not found on HuggingFace Hub, initialising classification head with random weights."
                    " You should TRAIN this model on a downstream task to use it for predictions and inference."
                )
                model_head_file = None

        model_card_data: SetFitModelCardData = model_kwargs.pop("model_card_data", SetFitModelCardData())

        if model_head_file is not None:
            model_head = joblib.load(model_head_file)
            if isinstance(model_head, torch.nn.Module):
                model_head.to(device)
            model_card_data.infer_st_id(model_id)
        else:
            head_params = model_kwargs.pop("head_params", {})
            if use_differentiable_head:
                if multi_target_strategy is None:
                    use_multitarget = False
                else:
                    if multi_target_strategy in ["one-vs-rest", "multi-output"]:
                        use_multitarget = True
                    else:
                        raise ValueError(
                            f"multi_target_strategy '{multi_target_strategy}' is not supported for differentiable head"
                        )
                # Base `model_head` parameters
                # - get the sentence embedding dimension from the `model_body`
                # - follow the `model_body`, put `model_head` on the target device
                base_head_params = {
                    "in_features": model_body.get_sentence_embedding_dimension(),
                    "device": device,
                    "multitarget": use_multitarget,
                }
                model_head = SetFitHead(**{**head_params, **base_head_params})
            else:
                clf = LogisticRegression(**head_params)
                if multi_target_strategy is not None:
                    if multi_target_strategy == "one-vs-rest":
                        multilabel_classifier = OneVsRestClassifier(clf)
                    elif multi_target_strategy == "multi-output":
                        multilabel_classifier = MultiOutputClassifier(clf)
                    elif multi_target_strategy == "classifier-chain":
                        multilabel_classifier = ClassifierChain(clf)
                    else:
                        raise ValueError(f"multi_target_strategy {multi_target_strategy} is not supported.")

                    model_head = multilabel_classifier
                else:
                    model_head = clf

            model_card_data.set_st_id(model_id if "/" in model_id else f"sentence-transformers/{model_id}")

        # Remove the `transformers` config
        model_kwargs.pop("config", None)
        return cls(
            model_body=model_body,
            model_head=model_head,
            multi_target_strategy=multi_target_strategy,
            model_card_data=model_card_data,
            sentence_transformers_kwargs=sentence_transformers_kwargs,
            **model_kwargs,
        )


docstring = SetFitModel.from_pretrained.__doc__
cut_index = docstring.find("model_kwargs")
if cut_index != -1:
    docstring = (
        docstring[:cut_index]
        + """labels (`List[str]`, *optional*):
                If the labels are integers ranging from `0` to `num_classes-1`, then these labels indicate
                    the corresponding labels.
            model_card_data (`SetFitModelCardData`, *optional*):
                A `SetFitModelCardData` instance storing data such as model language, license, dataset name,
                    etc. to be used in the automatically generated model cards.
            multi_target_strategy (`str`, *optional*):
                The strategy to use with multi-label classification. One of "one-vs-rest", "multi-output",
                    or "classifier-chain".
            use_differentiable_head (`bool`, *optional*):
                Whether to load SetFit using a differentiable (i.e., Torch) head instead of Logistic Regression.
            normalize_embeddings (`bool`, *optional*):
                Whether to apply normalization on the embeddings produced by the Sentence Transformer body.
            device (`Union[torch.device, str]`, *optional*):
                The device on which to load the SetFit model, e.g. `"cuda:0"`, `"mps"` or `torch.device("cuda")`.
            trust_remote_code (`bool`, defaults to `False`): Whether or not to allow for custom Sentence Transformers
                models defined on the Hub in their own modeling files. This option should only be set to True for
                repositories you trust and in which you have read the code, as it will execute code present on
                the Hub on your local machine. Defaults to False.

        Example::

            >>> from setfit import SetFitModel
            >>> model = SetFitModel.from_pretrained(
            ...     "sentence-transformers/paraphrase-mpnet-base-v2",
            ...     labels=["positive", "negative"],
            ... )
        """
    )
    SetFitModel.from_pretrained = set_docstring(SetFitModel.from_pretrained, docstring)

SetFitModel.save_pretrained = copy_func(SetFitModel.save_pretrained)
SetFitModel.save_pretrained.__doc__ = SetFitModel.save_pretrained.__doc__.replace(
    "~ModelHubMixin._from_pretrained", "SetFitModel.push_to_hub"
)
