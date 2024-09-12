import collections
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field, fields
from pathlib import Path
from platform import python_version
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import datasets
import tokenizers
import torch
import transformers
from datasets import Dataset
from huggingface_hub import CardData, ModelCard, dataset_info, list_datasets, model_info
from huggingface_hub.repocard_data import EvalResult, eval_results_to_model_index
from huggingface_hub.utils import yaml_dump
from sentence_transformers import __version__ as sentence_transformers_version
from transformers import PretrainedConfig, TrainerCallback
from transformers.integrations import CodeCarbonCallback
from transformers.modelcard import make_markdown_table
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from setfit import __version__ as setfit_version

from . import logging


logger = logging.get_logger(__name__)

if TYPE_CHECKING:
    from setfit.modeling import SetFitModel
    from setfit.trainer import Trainer


class ModelCardCallback(TrainerCallback):
    def __init__(self, trainer: "Trainer") -> None:
        super().__init__()
        self.trainer = trainer

        callbacks = [
            callback
            for callback in self.trainer.callback_handler.callbacks
            if isinstance(callback, CodeCarbonCallback)
        ]
        if callbacks:
            trainer.model.model_card_data.code_carbon_callback = callbacks[0]

    def on_init_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model: "SetFitModel", **kwargs
    ):
        if not model.model_card_data.dataset_id:
            # Inferring is hacky - it may break in the future, so let's be safe
            try:
                model.model_card_data.infer_dataset_id(self.trainer.train_dataset)
            except Exception:
                pass

        dataset = self.trainer.eval_dataset or self.trainer.train_dataset
        if dataset is not None:
            if not model.model_card_data.widget:
                model.model_card_data.set_widget_examples(dataset)

        if self.trainer.train_dataset:
            model.model_card_data.set_train_set_metrics(self.trainer.train_dataset)
            # Does not work for multilabel
            try:
                model.model_card_data.num_classes = len(set(self.trainer.train_dataset["label"]))
                model.model_card_data.set_label_examples(self.trainer.train_dataset)
            except Exception:
                pass

    def on_train_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model: "SetFitModel", **kwargs
    ) -> None:
        # model.model_card_data.hyperparameters = extract_hyperparameters_from_trainer(self.trainer)
        ignore_keys = {
            "output_dir",
            "logging_dir",
            "logging_strategy",
            "logging_first_step",
            "logging_steps",
            "eval_strategy",
            "eval_steps",
            "eval_delay",
            "save_strategy",
            "save_steps",
            "save_total_limit",
            "metric_for_best_model",
            "greater_is_better",
            "report_to",
            "samples_per_label",
            "show_progress_bar",
        }
        get_name_keys = {"loss", "distance_metric"}
        args_dict = args.to_dict()
        model.model_card_data.hyperparameters = {
            key: value.__name__ if key in get_name_keys else value
            for key, value in args_dict.items()
            if key not in ignore_keys and value is not None
        }

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: "SetFitModel",
        metrics: Dict[str, float],
        **kwargs,
    ) -> None:
        if (
            model.model_card_data.eval_lines_list
            and model.model_card_data.eval_lines_list[-1]["Step"] == state.global_step
        ):
            model.model_card_data.eval_lines_list[-1]["Validation Loss"] = metrics["eval_embedding_loss"]
        else:
            model.model_card_data.eval_lines_list.append(
                {
                    # "Training Loss": self.state.log_history[-1]["loss"] if "loss" in self.state.log_history[-1] else "-",
                    "Epoch": state.epoch,
                    "Step": state.global_step,
                    "Training Loss": "-",
                    "Validation Loss": metrics["eval_embedding_loss"],
                }
            )

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: "SetFitModel",
        logs: Dict[str, float],
        **kwargs,
    ):
        keys = {"embedding_loss", "polarity_embedding_loss", "aspect_embedding_loss"} & set(logs)
        if keys:
            if (
                model.model_card_data.eval_lines_list
                and model.model_card_data.eval_lines_list[-1]["Step"] == state.global_step
            ):
                model.model_card_data.eval_lines_list[-1]["Training Loss"] = logs[keys.pop()]
            else:
                model.model_card_data.eval_lines_list.append(
                    {
                        "Epoch": state.epoch,
                        "Step": state.global_step,
                        "Training Loss": logs[keys.pop()],
                        "Validation Loss": "-",
                    }
                )


YAML_FIELDS = [
    "language",
    "license",
    "library_name",
    "tags",
    "datasets",
    "metrics",
    "pipeline_tag",
    "widget",
    "model-index",
    "co2_eq_emissions",
    "base_model",
    "inference",
]
IGNORED_FIELDS = ["model"]


@dataclass
class SetFitModelCardData(CardData):
    """A dataclass storing data used in the model card.

    Args:
        language (`Optional[Union[str, List[str]]]`): The model language, either a string or a list,
            e.g. "en" or ["en", "de", "nl"]
        license (`Optional[str]`): The license of the model, e.g. "apache-2.0", "mit",
            or "cc-by-nc-sa-4.0"
        model_name (`Optional[str]`): The pretty name of the model, e.g. "SetFit with mBERT-base on SST2".
            If not defined, uses encoder_name/encoder_id and dataset_name/dataset_id to generate a model name.
        model_id (`Optional[str]`): The model ID when pushing the model to the Hub,
            e.g. "tomaarsen/span-marker-mbert-base-multinerd".
        dataset_name (`Optional[str]`): The pretty name of the dataset, e.g. "SST2".
        dataset_id (`Optional[str]`): The dataset ID of the dataset, e.g. "dair-ai/emotion".
        dataset_revision (`Optional[str]`): The dataset revision/commit that was for training/evaluation.
        st_id (`Optional[str]`): The Sentence Transformers model ID.

    <Tip>

    Install [``codecarbon``](https://github.com/mlco2/codecarbon) to automatically track carbon emission usage and
    include it in your model cards.

    </Tip>

    Example::

        >>> model = SetFitModel.from_pretrained(
        ...     "sentence-transformers/paraphrase-mpnet-base-v2",
        ...     labels=["negative", "positive"],
        ...     # Model card variables
        ...     model_card_data=SetFitModelCardData(
        ...         model_id="tomaarsen/setfit-paraphrase-mpnet-base-v2-sst2",
        ...         dataset_name="SST2",
        ...         dataset_id="sst2",
        ...         license="apache-2.0",
        ...         language="en",
        ...     ),
        ... )
    """

    # Potentially provided by the user
    language: Optional[Union[str, List[str]]] = None
    license: Optional[str] = None
    tags: Optional[List[str]] = field(
        default_factory=lambda: [
            "setfit",
            "sentence-transformers",
            "text-classification",
            "generated_from_setfit_trainer",
        ]
    )
    model_name: Optional[str] = None
    model_id: Optional[str] = None
    dataset_name: Optional[str] = None
    dataset_id: Optional[str] = None
    dataset_revision: Optional[str] = None
    task_name: Optional[str] = None
    st_id: Optional[str] = None

    # Automatically filled by `ModelCardCallback` and the Trainer directly
    hyperparameters: Dict[str, Any] = field(default_factory=dict, init=False)
    eval_results_dict: Optional[Dict[str, Any]] = field(default_factory=dict, init=False)
    eval_lines_list: List[Dict[str, float]] = field(default_factory=list, init=False)
    metric_lines: List[Dict[str, float]] = field(default_factory=list, init=False)
    widget: List[Dict[str, str]] = field(default_factory=list, init=False)
    predict_example: Optional[str] = field(default=None, init=False)
    label_example_list: List[Dict[str, str]] = field(default_factory=list, init=False)
    tokenizer_warning: bool = field(default=False, init=False)
    train_set_metrics_list: List[Dict[str, str]] = field(default_factory=list, init=False)
    train_set_sentences_per_label_list: List[Dict[str, str]] = field(default_factory=list, init=False)
    code_carbon_callback: Optional[CodeCarbonCallback] = field(default=None, init=False)
    num_classes: Optional[int] = field(default=None, init=False)
    best_model_step: Optional[int] = field(default=None, init=False)
    metrics: List[str] = field(default_factory=lambda: ["accuracy"], init=False)

    # Computed once, always unchanged
    pipeline_tag: str = field(default="text-classification", init=False)
    library_name: str = field(default="setfit", init=False)
    version: Dict[str, str] = field(
        default_factory=lambda: {
            "python": python_version(),
            "setfit": setfit_version,
            "sentence_transformers": sentence_transformers_version,
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "datasets": datasets.__version__,
            "tokenizers": tokenizers.__version__,
        },
        init=False,
    )

    # ABSA-related arguments
    absa: Dict[str, Any] = field(default=None, init=False, repr=False)

    # Passed via `register_model` only
    model: Optional["SetFitModel"] = field(default=None, init=False, repr=False)
    head_class: Optional[str] = field(default=None, init=False, repr=False)
    inference: Optional[bool] = field(default=True, init=False, repr=False)

    def __post_init__(self):
        # We don't want to save "ignore_metadata_errors" in our Model Card
        if self.dataset_id:
            if is_on_huggingface(self.dataset_id, is_model=False):
                if self.language is None:
                    # if languages are not set, try to determine the language from the dataset on the Hub
                    try:
                        info = dataset_info(self.dataset_id)
                    except Exception:
                        pass
                    else:
                        if info.cardData:
                            self.language = info.cardData.get("language", self.language)
            else:
                logger.warning(
                    f"The provided {self.dataset_id!r} dataset could not be found on the Hugging Face Hub."
                    " Setting `dataset_id` to None."
                )
                self.dataset_id = None

        if self.model_id and self.model_id.count("/") != 1:
            logger.warning(
                f"The provided {self.model_id!r} model ID should include the organization or user,"
                ' such as "tomaarsen/setfit-bge-small-v1.5-sst2-8-shot". Setting `model_id` to None.'
            )
            self.model_id = None

    def set_best_model_step(self, step: int) -> None:
        self.best_model_step = step

    def set_widget_examples(self, dataset: Dataset) -> None:
        samples = dataset.select(random.sample(range(len(dataset)), k=min(len(dataset), 5)))["text"]
        self.widget = [{"text": sample} for sample in samples]

        samples.sort(key=len)
        if samples:
            self.predict_example = samples[0]

    def set_train_set_metrics(self, dataset: Dataset) -> None:
        def add_naive_word_count(sample: Dict[str, Any]) -> Dict[str, Any]:
            sample["word_count"] = len(sample["text"].split(" "))
            return sample

        dataset = dataset.map(add_naive_word_count)
        self.train_set_metrics_list = [
            {
                "Training set": "Word count",
                "Min": min(dataset["word_count"]),
                "Median": sum(dataset["word_count"]) / len(dataset),
                "Max": max(dataset["word_count"]),
            },
        ]
        # E.g. if unlabeled via DistillationTrainer
        if "label" not in dataset.column_names:
            return

        sample_label = dataset[0]["label"]
        if isinstance(sample_label, collections.abc.Sequence) and not isinstance(sample_label, str):
            return
        try:
            counter = Counter(dataset["label"])
            if self.model.labels:
                self.train_set_sentences_per_label_list = [
                    {
                        "Label": str_label,
                        "Training Sample Count": counter[
                            str_label if isinstance(sample_label, str) else self.model.label2id[str_label]
                        ],
                    }
                    for str_label in self.model.labels
                ]
            else:
                self.train_set_sentences_per_label_list = [
                    {
                        "Label": (
                            self.model.labels[label] if self.model.labels and isinstance(label, int) else str(label)
                        ),
                        "Training Sample Count": count,
                    }
                    for label, count in sorted(counter.items())
                ]
        except Exception:
            # There are some tricky edge cases possible, e.g. if the user provided integer labels that do not fall
            # between 0 to num_classes-1, so we make sure we never cause errors.
            pass

    def set_label_examples(self, dataset: Dataset) -> None:
        num_examples_per_label = 3
        examples = defaultdict(list)
        finished_labels = set()
        for sample in dataset:
            text = sample["text"]
            label = sample["label"]
            if label not in finished_labels:
                examples[label].append(f"<li>{repr(text)}</li>")
                if len(examples[label]) >= num_examples_per_label:
                    finished_labels.add(label)
            if len(finished_labels) == self.num_classes:
                break
        self.label_example_list = [
            {
                "Label": self.model.labels[label] if self.model.labels and isinstance(label, int) else label,
                "Examples": "<ul>" + "".join(example_set) + "</ul>",
            }
            for label, example_set in examples.items()
        ]

    def infer_dataset_id(self, dataset: Dataset) -> None:
        def subtuple_finder(tuple: Tuple[str], subtuple: Tuple[str]) -> int:
            for i, element in enumerate(tuple):
                if element == subtuple[0] and tuple[i : i + len(subtuple)] == subtuple:
                    return i
            return -1

        def normalize(dataset_id: str) -> str:
            for token in "/\\_-":
                dataset_id = dataset_id.replace(token, "")
            return dataset_id.lower()

        cache_files = dataset.cache_files
        if cache_files and "filename" in cache_files[0]:
            cache_path_parts = Path(cache_files[0]["filename"]).parts
            # Check if the cachefile is under "huggingface/datasets"
            subtuple = ("huggingface", "datasets")
            index = subtuple_finder(cache_path_parts, subtuple)
            if index == -1:
                return

            # Get the folder after "huggingface/datasets"
            cache_dataset_name = cache_path_parts[index + len(subtuple)]
            # If the dataset has an author:
            if "___" in cache_dataset_name:
                author, dataset_name = cache_dataset_name.split("___")
            else:
                author = None
                dataset_name = cache_dataset_name

            # Make sure the normalized dataset IDs match
            dataset_list = [
                dataset
                for dataset in list_datasets(author=author, dataset_name=dataset_name)
                if normalize(dataset.id) == normalize(cache_dataset_name)
            ]
            # If there's only one match, get the ID from it
            if len(dataset_list) == 1:
                self.dataset_id = dataset_list[0].id

    def register_model(self, model: "SetFitModel") -> None:
        self.model = model
        head_class = model.model_head.__class__.__name__
        self.head_class = {
            "LogisticRegression": "[LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)",
            "SetFitHead": "[SetFitHead](huggingface.co/docs/setfit/reference/main#setfit.SetFitHead)",
        }.get(head_class, head_class)

        if not self.model_name:
            if self.st_id:
                self.model_name = f"SetFit with {self.st_id}"
                if self.dataset_name or self.dataset_id:
                    self.model_name += f" on {self.dataset_name or self.dataset_id}"
            else:
                self.model_name = "SetFit"

        self.inference = self.model.multi_target_strategy is None

    def infer_st_id(self, setfit_model_id: str) -> None:
        config_dict, _ = PretrainedConfig.get_config_dict(setfit_model_id)
        st_id = config_dict.get("_name_or_path")
        st_id_path = Path(st_id)
        # Sometimes the name_or_path ends exactly with the model_id, e.g.
        # "C:\\Users\\tom/.cache\\torch\\sentence_transformers\\BAAI_bge-small-en-v1.5\\"
        candidate_model_ids = ["/".join(st_id_path.parts[-2:])]
        # Sometimes the name_or_path its final part contains the full model_id, with "/" replaced with a "_", e.g.
        # "/root/.cache/torch/sentence_transformers/sentence-transformers_all-mpnet-base-v2/"
        # In that case, we take the last part, split on _, and try all combinations
        # e.g. "a_b_c_d" -> ['a/b_c_d', 'a_b/c_d', 'a_b_c/d']
        splits = st_id_path.name.split("_")
        candidate_model_ids += ["_".join(splits[:idx]) + "/" + "_".join(splits[idx:]) for idx in range(1, len(splits))]
        for model_id in candidate_model_ids:
            if is_on_huggingface(model_id):
                self.st_id = model_id
                break

    def set_st_id(self, model_id: str) -> None:
        if is_on_huggingface(model_id):
            self.st_id = model_id

    def post_training_eval_results(self, results: Dict[str, float]) -> None:
        def try_to_pure_python(value: Any) -> Any:
            """Try to convert a value from a Numpy or Torch scalar to pure Python, if not already pure Python"""
            try:
                if hasattr(value, "dtype"):
                    return value.item()
            except Exception:
                pass
            return value

        pure_python_results = {key: try_to_pure_python(value) for key, value in results.items()}
        results_without_split = {
            key.split("_", maxsplit=1)[1].title(): value for key, value in pure_python_results.items()
        }
        self.eval_results_dict = pure_python_results
        self.metric_lines = [{"Label": "**all**", **results_without_split}]

    def _maybe_round(self, v, decimals=4):
        if isinstance(v, float) and len(str(v).split(".")) > 1 and len(str(v).split(".")[1]) > decimals:
            return f"{v:.{decimals}f}"
        return str(v)

    def to_dict(self) -> Dict[str, Any]:
        super_dict = {field.name: getattr(self, field.name) for field in fields(self)}

        # Compute required formats from the raw data
        if self.eval_results_dict:
            dataset_split = list(self.eval_results_dict.keys())[0].split("_")[0]
            dataset_id = self.dataset_id or "unknown"
            dataset_name = self.dataset_name or self.dataset_id or "Unknown"
            eval_results = [
                EvalResult(
                    task_type="text-classification",
                    dataset_type=dataset_id,
                    dataset_name=dataset_name,
                    dataset_split=dataset_split,
                    dataset_revision=self.dataset_revision,
                    metric_type=metric_key.split("_", maxsplit=1)[1],
                    metric_value=metric_value,
                    task_name="Text Classification",
                    metric_name=metric_key.split("_", maxsplit=1)[1].title(),
                )
                for metric_key, metric_value in self.eval_results_dict.items()
            ]
            super_dict["metrics"] = [metric_key.split("_", maxsplit=1)[1] for metric_key in self.eval_results_dict]
            super_dict["model-index"] = eval_results_to_model_index(self.model_name, eval_results)
        eval_lines_list = [
            {
                key: f"**{self._maybe_round(value)}**" if line["Step"] == self.best_model_step else value
                for key, value in line.items()
            }
            for line in self.eval_lines_list
        ]
        super_dict["eval_lines"] = make_markdown_table(eval_lines_list)
        super_dict["explain_bold_in_eval"] = "**" in super_dict["eval_lines"]
        # Replace |:---:| with |:---| for left alignment
        super_dict["label_examples"] = make_markdown_table(self.label_example_list).replace("-:|", "--|")
        super_dict["train_set_metrics"] = make_markdown_table(self.train_set_metrics_list).replace("-:|", "--|")
        super_dict["train_set_sentences_per_label_list"] = make_markdown_table(
            self.train_set_sentences_per_label_list
        ).replace("-:|", "--|")
        super_dict["metrics_table"] = make_markdown_table(self.metric_lines).replace("-:|", "--|")
        if self.code_carbon_callback and self.code_carbon_callback.tracker:
            emissions_data = self.code_carbon_callback.tracker._prepare_emissions_data()
            super_dict["co2_eq_emissions"] = {
                # * 1000 to convert kg to g
                "emissions": float(emissions_data.emissions) * 1000,
                "source": "codecarbon",
                "training_type": "fine-tuning",
                "on_cloud": emissions_data.on_cloud == "Y",
                "cpu_model": emissions_data.cpu_model,
                "ram_total_size": emissions_data.ram_total_size,
                "hours_used": round(emissions_data.duration / 3600, 3),
            }
            if emissions_data.gpu_model:
                super_dict["co2_eq_emissions"]["hardware_used"] = emissions_data.gpu_model
        if self.dataset_id:
            super_dict["datasets"] = [self.dataset_id]
        if self.st_id:
            super_dict["base_model"] = self.st_id
        super_dict["model_max_length"] = self.model.model_body.get_max_seq_length()
        if super_dict["num_classes"] is None:
            if self.model.labels:
                super_dict["num_classes"] = len(self.model.labels)
        if super_dict["absa"]:
            super_dict.update(super_dict.pop("absa"))

        for key in IGNORED_FIELDS:
            super_dict.pop(key, None)
        return super_dict

    def to_yaml(self, line_break=None) -> str:
        return yaml_dump(
            {key: value for key, value in self.to_dict().items() if key in YAML_FIELDS and value is not None},
            sort_keys=False,
            line_break=line_break,
        ).strip()


def is_on_huggingface(repo_id: str, is_model: bool = True) -> bool:
    # Models with more than two 'sections' certainly are not public models
    if len(repo_id.split("/")) > 2:
        return False

    try:
        if is_model:
            model_info(repo_id)
        else:
            dataset_info(repo_id)
        return True
    except Exception:
        # Fetching models can fail for many reasons: Repository not existing, no internet access, HF down, etc.
        return False


def generate_model_card(model: "SetFitModel") -> str:
    template_path = Path(__file__).parent / "model_card_template.md"
    model_card = ModelCard.from_template(card_data=model.model_card_data, template_path=template_path, hf_emoji="🤗")
    return model_card.content
