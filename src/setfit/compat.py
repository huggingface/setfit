"""Imports that differ between the supported versions of transformers and Sentence Transformers."""

from packaging.version import Version, parse
from sentence_transformers import __version__ as _sentence_transformers_version
from transformers import __version__ as _transformers_version


TRANSFORMERS_VERSION: Version = parse(_transformers_version)
SENTENCE_TRANSFORMERS_VERSION: Version = parse(_sentence_transformers_version)

# Sentence Transformers v5.4 moved these modules and deprecated the old import paths
try:
    from sentence_transformers.sentence_transformer import losses
    from sentence_transformers.sentence_transformer.model_card import SentenceTransformerModelCardCallback
    from sentence_transformers.sentence_transformer.modules import Dense
    from sentence_transformers.sentence_transformer.training_args import BatchSamplers
except ImportError:
    from sentence_transformers import losses
    from sentence_transformers.models import Dense
    from sentence_transformers.training_args import BatchSamplers

    try:
        from sentence_transformers.model_card import SentenceTransformerModelCardCallback
    except ImportError:
        # Sentence Transformers < 4.0
        from sentence_transformers.model_card import ModelCardCallback as SentenceTransformerModelCardCallback

# transformers v5 moved default_logdir
try:
    from transformers.integrations.integration_utils import default_logdir
except ImportError:
    from transformers.training_args import default_logdir


__all__ = [
    "BatchSamplers",
    "Dense",
    "SENTENCE_TRANSFORMERS_VERSION",
    "SentenceTransformerModelCardCallback",
    "TRANSFORMERS_VERSION",
    "Version",
    "default_logdir",
    "losses",
]
