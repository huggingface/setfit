from dataclasses import dataclass
from typing import Any, Optional

from transformers import PretrainedConfig


@dataclass
class SetFitModelConfig:
    """
    Config for SetFitModel.

    Parameters:
        model_body (`PretrainedConfig`):
            Config of the model_body transformer.
        model_head (`Optional[Any]`):
            Config of the model_head.
    """

    model_body: PretrainedConfig
    model_head: Optional[Any]
