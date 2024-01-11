import copy
import os
import re
import tempfile
import types
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union

import torch
from datasets import Dataset
from huggingface_hub.utils import SoftTemporaryDirectory

from setfit.utils import set_docstring

from .. import logging
from ..modeling import SetFitModel
from .aspect_extractor import AspectExtractor


if TYPE_CHECKING:
    from spacy.tokens import Doc

logger = logging.get_logger(__name__)


@dataclass
class SpanSetFitModel(SetFitModel):
    spacy_model: str = "en_core_web_lg"
    span_context: int = 0

    attributes_to_save: Set[str] = field(
        init=False,
        repr=False,
        default_factory=lambda: {"normalize_embeddings", "labels", "span_context", "spacy_model"},
    )

    def prepend_aspects(self, docs: List["Doc"], aspects_list: List[List[slice]]) -> List[str]:
        for doc, aspects in zip(docs, aspects_list):
            for aspect_slice in aspects:
                aspect = doc[max(aspect_slice.start - self.span_context, 0) : aspect_slice.stop + self.span_context]
                # TODO: Investigate performance difference of different formats
                yield aspect.text + ":" + doc.text

    def __call__(self, docs: List["Doc"], aspects_list: List[List[slice]]) -> List[bool]:
        inputs_list = list(self.prepend_aspects(docs, aspects_list))
        preds = self.predict(inputs_list, as_numpy=True)
        iter_preds = iter(preds)
        return [[next(iter_preds) for _ in aspects] for aspects in aspects_list]

    def create_model_card(self, path: str, model_name: Optional[str] = None) -> None:
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
            model_name = "/".join(model_path.parts[-2:])

        is_aspect = isinstance(self, AspectModel)
        aspect_model = "setfit-absa-aspect"
        polarity_model = "setfit-absa-polarity"
        if model_name is not None:
            if is_aspect:
                aspect_model = model_name
                if model_name.endswith("-aspect"):
                    polarity_model = model_name[: -len("-aspect")] + "-polarity"
            else:
                polarity_model = model_name
                if model_name.endswith("-polarity"):
                    aspect_model = model_name[: -len("-polarity")] + "-aspect"

        # Only once:
        if self.model_card_data.absa is None and self.model_card_data.model_name:
            from spacy import __version__ as spacy_version

            self.model_card_data.model_name = self.model_card_data.model_name.replace(
                "SetFit", "SetFit Aspect Model" if is_aspect else "SetFit Polarity Model", 1
            )
            self.model_card_data.tags.insert(1, "absa")
            self.model_card_data.version["spacy"] = spacy_version
        self.model_card_data.absa = {
            "is_absa": True,
            "is_aspect": is_aspect,
            "spacy_model": self.spacy_model,
            "aspect_model": aspect_model,
            "polarity_model": polarity_model,
        }
        if self.model_card_data.task_name is None:
            self.model_card_data.task_name = "Aspect Based Sentiment Analysis (ABSA)"
        self.model_card_data.inference = False
        with open(os.path.join(path, "README.md"), "w", encoding="utf-8") as f:
            f.write(self.generate_model_card())


docstring = SpanSetFitModel.from_pretrained.__doc__
cut_index = docstring.find("multi_target_strategy")
if cut_index != -1:
    docstring = (
        docstring[:cut_index]
        + """model_card_data (`SetFitModelCardData`, *optional*):
                A `SetFitModelCardData` instance storing data such as model language, license, dataset name,
                    etc. to be used in the automatically generated model cards.
            use_differentiable_head (`bool`, *optional*):
                Whether to load SetFit using a differentiable (i.e., Torch) head instead of Logistic Regression.
            normalize_embeddings (`bool`, *optional*):
                Whether to apply normalization on the embeddings produced by the Sentence Transformer body.
            span_context (`int`, defaults to `0`):
                The number of words before and after the span candidate that should be prepended to the full sentence.
                By default, 0 for Aspect models and 3 for Polarity models.
            device (`Union[torch.device, str]`, *optional*):
                The device on which to load the SetFit model, e.g. `"cuda:0"`, `"mps"` or `torch.device("cuda")`."""
    )
    SpanSetFitModel.from_pretrained = set_docstring(SpanSetFitModel.from_pretrained, docstring, cls=SpanSetFitModel)


class AspectModel(SpanSetFitModel):
    def __call__(self, docs: List["Doc"], aspects_list: List[List[slice]]) -> List[bool]:
        sentence_preds = super().__call__(docs, aspects_list)
        return [
            [aspect for aspect, pred in zip(aspects, preds) if pred == "aspect"]
            for aspects, preds in zip(aspects_list, sentence_preds)
        ]


# The set_docstring magic has as a consequences that subclasses need to update the cls in the from_pretrained
# classmethod, otherwise the wrong instance will be instantiated.
AspectModel.from_pretrained = types.MethodType(AspectModel.from_pretrained.__func__, AspectModel)


@dataclass
class PolarityModel(SpanSetFitModel):
    span_context: int = 3


PolarityModel.from_pretrained = types.MethodType(PolarityModel.from_pretrained.__func__, PolarityModel)


@dataclass
class AbsaModel:
    aspect_extractor: AspectExtractor
    aspect_model: AspectModel
    polarity_model: PolarityModel

    def gold_aspect_spans_to_aspects_list(self, inputs: Dataset) -> List[List[slice]]:
        # First group inputs by text
        grouped_data = defaultdict(list)
        for sample in inputs:
            text = sample.pop("text")
            grouped_data[text].append(sample)

        # Get the spaCy docs
        docs, _ = self.aspect_extractor(grouped_data.keys())

        # Get the aspect spans for each doc by matching gold spans to the spaCy tokens
        aspects_list = []
        index = -1
        skipped_indices = []
        for doc, samples in zip(docs, grouped_data.values()):
            aspects_list.append([])
            for sample in samples:
                index += 1
                match_objects = re.finditer(re.escape(sample["span"]), doc.text)
                for i, match in enumerate(match_objects):
                    if i == sample["ordinal"]:
                        char_idx_start = match.start()
                        char_idx_end = match.end()
                        span = doc.char_span(char_idx_start, char_idx_end)
                        if span is None:
                            logger.warning(
                                f"Aspect term {sample['span']!r} with ordinal {sample['ordinal']}, isn't a token in {doc.text!r} according to spaCy. "
                                "Skipping this sample."
                            )
                            skipped_indices.append(index)
                            continue
                        aspects_list[-1].append(slice(span.start, span.end))
        return docs, aspects_list, skipped_indices

    def predict_dataset(self, inputs: Dataset) -> Dataset:
        if set(inputs.column_names) >= {"text", "span", "ordinal"}:
            pass
        elif set(inputs.column_names) >= {"text", "span"}:
            inputs = inputs.add_column("ordinal", [0] * len(inputs))
        else:
            raise ValueError(
                "`inputs` must be either a `str`, a `List[str]`, or a `datasets.Dataset` with columns `text` and `span` and optionally `ordinal`. "
                f"Found a dataset with these columns: {inputs.column_names}."
            )
        if "pred_polarity" in inputs.column_names:
            raise ValueError(
                "`predict_dataset` wants to add a `pred_polarity` column, but the input dataset already contains that column."
            )
        docs, aspects_list, skipped_indices = self.gold_aspect_spans_to_aspects_list(inputs)
        polarity_list = sum(self.polarity_model(docs, aspects_list), [])
        for index in skipped_indices:
            polarity_list.insert(index, None)
        return inputs.add_column("pred_polarity", polarity_list)

    def predict(self, inputs: Union[str, List[str], Dataset]) -> Union[List[Dict[str, Any]], Dataset]:
        """Predicts aspects & their polarities of the given inputs.

        Example::

            >>> from setfit import AbsaModel
            >>> model = AbsaModel.from_pretrained(
            ...     "tomaarsen/setfit-absa-bge-small-en-v1.5-restaurants-aspect",
            ...     "tomaarsen/setfit-absa-bge-small-en-v1.5-restaurants-polarity",
            ... )
            >>> model.predict("The food and wine are just exquisite.")
            [{'span': 'food', 'polarity': 'positive'}, {'span': 'wine', 'polarity': 'positive'}]

            >>> from setfit import AbsaModel
            >>> from datasets import load_dataset
            >>> model = AbsaModel.from_pretrained(
            ...     "tomaarsen/setfit-absa-bge-small-en-v1.5-restaurants-aspect",
            ...     "tomaarsen/setfit-absa-bge-small-en-v1.5-restaurants-polarity",
            ... )
            >>> dataset = load_dataset("tomaarsen/setfit-absa-semeval-restaurants", split="train")
            >>> model.predict(dataset)
            Dataset({
                features: ['text', 'span', 'label', 'ordinal', 'pred_polarity'],
                num_rows: 3693
            })

        Args:
            inputs (Union[str, List[str], Dataset]): Either a sentence, a list of sentences,
                or a dataset with columns `text` and `span` and optionally `ordinal`. This dataset
                contains gold aspects, and we only predict the polarities for them.

        Returns:
            Union[List[Dict[str, Any]], Dataset]: Either a list of dictionaries with keys `span`
                and `polarity` if the input was a sentence or a list of sentences, or a dataset with
                columns `text`, `span`, `ordinal`, and `pred_polarity`.
        """
        if isinstance(inputs, Dataset):
            return self.predict_dataset(inputs)

        is_str = isinstance(inputs, str)
        inputs_list = [inputs] if is_str else inputs
        docs, aspects_list = self.aspect_extractor(inputs_list)
        if sum(aspects_list, []) == []:
            return aspects_list

        aspects_list = self.aspect_model(docs, aspects_list)
        if sum(aspects_list, []) == []:
            return aspects_list

        polarity_list = self.polarity_model(docs, aspects_list)
        outputs = []
        for docs, aspects, polarities in zip(docs, aspects_list, polarity_list):
            outputs.append(
                [
                    {"span": docs[aspect_slice].text, "polarity": polarity}
                    for aspect_slice, polarity in zip(aspects, polarities)
                ]
            )
        return outputs if not is_str else outputs[0]

    @property
    def device(self) -> torch.device:
        return self.aspect_model.device

    def to(self, device: Union[str, torch.device]) -> "AbsaModel":
        self.aspect_model.to(device)
        self.polarity_model.to(device)

    def __call__(self, inputs: Union[str, List[str]]) -> List[Dict[str, Any]]:
        return self.predict(inputs)

    def save_pretrained(
        self,
        save_directory: Union[str, Path],
        polarity_save_directory: Optional[Union[str, Path]] = None,
        push_to_hub: bool = False,
        **kwargs,
    ) -> None:
        if polarity_save_directory is None:
            base_save_directory = Path(save_directory)
            save_directory = base_save_directory.parent / (base_save_directory.name + "-aspect")
            polarity_save_directory = base_save_directory.parent / (base_save_directory.name + "-polarity")
        self.aspect_model.save_pretrained(save_directory=save_directory, push_to_hub=push_to_hub, **kwargs)
        self.polarity_model.save_pretrained(save_directory=polarity_save_directory, push_to_hub=push_to_hub, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        polarity_model_id: Optional[str] = None,
        spacy_model: Optional[str] = None,
        span_contexts: Tuple[Optional[int], Optional[int]] = (None, None),
        force_download: bool = None,
        resume_download: bool = None,
        proxies: Optional[Dict] = None,
        token: Optional[Union[str, bool]] = None,
        cache_dir: Optional[str] = None,
        local_files_only: bool = None,
        use_differentiable_head: bool = None,
        normalize_embeddings: bool = None,
        **model_kwargs,
    ) -> "AbsaModel":
        revision = None
        if len(model_id.split("@")) == 2:
            model_id, revision = model_id.split("@")
        if spacy_model:
            model_kwargs["spacy_model"] = spacy_model
        aspect_model = AspectModel.from_pretrained(
            model_id,
            span_context=span_contexts[0],
            revision=revision,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            use_differentiable_head=use_differentiable_head,
            normalize_embeddings=normalize_embeddings,
            labels=["no aspect", "aspect"],
            **model_kwargs,
        )
        if polarity_model_id:
            model_id = polarity_model_id
            revision = None
            if len(model_id.split("@")) == 2:
                model_id, revision = model_id.split("@")
        # If model_card_data was provided, "separate" the instance between the Aspect
        # and Polarity models.
        model_card_data = model_kwargs.pop("model_card_data", None)
        if model_card_data:
            model_kwargs["model_card_data"] = copy.deepcopy(model_card_data)
        polarity_model = PolarityModel.from_pretrained(
            model_id,
            span_context=span_contexts[1],
            revision=revision,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            use_differentiable_head=use_differentiable_head,
            normalize_embeddings=normalize_embeddings,
            **model_kwargs,
        )
        if aspect_model.spacy_model != polarity_model.spacy_model:
            logger.warning(
                "The Aspect and Polarity models are configured to use different spaCy models:\n"
                f"* {repr(aspect_model.spacy_model)} for the aspect model, and\n"
                f"* {repr(polarity_model.spacy_model)} for the polarity model.\n"
                f"This model will use {repr(aspect_model.spacy_model)}."
            )

        aspect_extractor = AspectExtractor(spacy_model=aspect_model.spacy_model)

        return cls(aspect_extractor, aspect_model, polarity_model)

    def push_to_hub(self, repo_id: str, polarity_repo_id: Optional[str] = None, **kwargs) -> None:
        if "/" not in repo_id:
            raise ValueError(
                '`repo_id` must be a full repository ID, including organisation, e.g. "tomaarsen/setfit-absa-restaurant".'
            )
        if polarity_repo_id is not None and "/" not in polarity_repo_id:
            raise ValueError(
                '`polarity_repo_id` must be a full repository ID, including organisation, e.g. "tomaarsen/setfit-absa-restaurant".'
            )
        commit_message = kwargs.pop("commit_message", "Add SetFit ABSA model")

        # Push the files to the repo in a single commit
        with SoftTemporaryDirectory() as tmp_dir:
            save_directory = Path(tmp_dir) / repo_id
            polarity_save_directory = None if polarity_repo_id is None else Path(tmp_dir) / polarity_repo_id
            self.save_pretrained(
                save_directory=save_directory,
                polarity_save_directory=polarity_save_directory,
                push_to_hub=True,
                commit_message=commit_message,
                **kwargs,
            )
