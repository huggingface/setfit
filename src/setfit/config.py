from dataclasses import dataclass
from typing import Any, Optional

from transformers import PretrainedConfig


@dataclass
class SetFitConfig:
    """
    Configuration for SetFitModel.

    Args:
        model_body (`PretrainedConfig`):
            Configuration of the Sentence Transformer body.
        model_head (`Optional[Any]`):
            Configuration of the head.
    """

    model_body: PretrainedConfig
    model_head: Optional[Any]
