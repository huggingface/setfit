from itertools import zip_longest
from typing import Generator, Iterable, List, Optional

import numpy as np
import torch
from sentence_transformers import InputExample
from torch.utils.data import IterableDataset

from . import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def shuffle_combinations(iterable: Iterable, replacement: bool = True) -> Generator:
    """Generates shuffled pair combinations for any iterable data provided.

    Args:
        iterable: data to generate pair combinations from
        replacement: enable to include combinations of same samples,
            equivalent to itertools.combinations_with_replacement

    Returns:
        Generator of shuffled pairs as a tuple
    """
    n = len(iterable)
    k = 1 if not replacement else 0
    idxs = np.stack(np.triu_indices(n, k), axis=-1)
    for i in np.random.RandomState(seed=42).permutation(len(idxs)):
        _idx, idx = idxs[i, :]
        yield iterable[_idx], iterable[idx]


class ContrastiveDataset(IterableDataset):
    def __init__(
        self,
        examples: List[InputExample],
        multilabel: bool,
        num_iterations: Optional[None] = None,
        sampling_strategy: str = "oversampling",
        max_pairs: int = -1,
    ) -> None:
        """Generates positive and negative text pairs for contrastive learning.

        Args:
            examples (InputExample): text and labels in a text transformer dataclass
            multilabel: set to process "multilabel" labels array
            sampling_strategy: "unique", "oversampling", or "undersampling"
            num_iterations: if provided explicitly sets the number of pairs to be generated
                where n_pairs = n_iterations * n_sentences * 2 (for pos & neg pairs)
            max_pairs: If not -1, then we only sample pairs until we have certainly reached
                max_pairs pairs.
        """
        super().__init__()
        self.pos_index = 0
        self.neg_index = 0
        self.pos_pairs = []
        self.neg_pairs = []
        self.sentences = np.array([s.texts[0] for s in examples])
        self.labels = np.array([s.label for s in examples])
        self.sentence_labels = list(zip(self.sentences, self.labels))
        self.max_pairs = max_pairs

        if multilabel:
            self.generate_multilabel_pairs()
        else:
            self.generate_pairs()

        if num_iterations is not None and num_iterations > 0:
            self.len_pos_pairs = num_iterations * len(self.sentences)
            self.len_neg_pairs = num_iterations * len(self.sentences)

        elif sampling_strategy == "unique":
            self.len_pos_pairs = len(self.pos_pairs)
            self.len_neg_pairs = len(self.neg_pairs)

        elif sampling_strategy == "undersampling":
            self.len_pos_pairs = min(len(self.pos_pairs), len(self.neg_pairs))
            self.len_neg_pairs = min(len(self.pos_pairs), len(self.neg_pairs))

        elif sampling_strategy == "oversampling":
            self.len_pos_pairs = max(len(self.pos_pairs), len(self.neg_pairs))
            self.len_neg_pairs = max(len(self.pos_pairs), len(self.neg_pairs))

        else:
            raise ValueError("Invalid sampling strategy. Must be one of 'unique', 'oversampling', or 'undersampling'.")

    def generate_pairs(self) -> None:
        for (_text, _label), (text, label) in shuffle_combinations(self.sentence_labels):
            if _label == label:
                self.pos_pairs.append(InputExample(texts=[_text, text], label=1.0))
            else:
                self.neg_pairs.append(InputExample(texts=[_text, text], label=0.0))
            if self.max_pairs != -1 and len(self.pos_pairs) > self.max_pairs and len(self.neg_pairs) > self.max_pairs:
                break

    def generate_multilabel_pairs(self) -> None:
        for (_text, _label), (text, label) in shuffle_combinations(self.sentence_labels):
            if any(np.logical_and(_label, label)):
                # logical_and checks if labels are both set for each class
                self.pos_pairs.append(InputExample(texts=[_text, text], label=1.0))
            else:
                self.neg_pairs.append(InputExample(texts=[_text, text], label=0.0))
            if self.max_pairs != -1 and len(self.pos_pairs) > self.max_pairs and len(self.neg_pairs) > self.max_pairs:
                break

    def get_positive_pairs(self) -> List[InputExample]:
        pairs = []
        for _ in range(self.len_pos_pairs):
            if self.pos_index >= len(self.pos_pairs):
                self.pos_index = 0
            pairs.append(self.pos_pairs[self.pos_index])
            self.pos_index += 1
        return pairs

    def get_negative_pairs(self) -> List[InputExample]:
        pairs = []
        for _ in range(self.len_neg_pairs):
            if self.neg_index >= len(self.neg_pairs):
                self.neg_index = 0
            pairs.append(self.neg_pairs[self.neg_index])
            self.neg_index += 1
        return pairs

    def __iter__(self):
        for pos_pair, neg_pair in zip_longest(self.get_positive_pairs(), self.get_negative_pairs()):
            if pos_pair is not None:
                yield pos_pair
            if neg_pair is not None:
                yield neg_pair

    def __len__(self) -> int:
        return self.len_pos_pairs + self.len_neg_pairs


class ContrastiveDistillationDataset(ContrastiveDataset):
    def __init__(
        self,
        examples: List[InputExample],
        cos_sim_matrix: torch.Tensor,
        num_iterations: Optional[None] = None,
        sampling_strategy: str = "oversampling",
        max_pairs: int = -1,
    ) -> None:
        self.cos_sim_matrix = cos_sim_matrix
        super().__init__(
            examples,
            multilabel=False,
            num_iterations=num_iterations,
            sampling_strategy=sampling_strategy,
            max_pairs=max_pairs,
        )
        # Internally we store all pairs in pos_pairs, regardless of sampling strategy.
        # After all, without labels, there isn't much of a strategy.
        self.sentence_labels = list(enumerate(self.sentences))

        self.len_neg_pairs = 0
        if num_iterations is not None and num_iterations > 0:
            self.len_pos_pairs = num_iterations * len(self.sentences)
        else:
            self.len_pos_pairs = len(self.pos_pairs)

    def generate_pairs(self) -> None:
        for (text_one, id_one), (text_two, id_two) in shuffle_combinations(self.sentence_labels):
            self.pos_pairs.append(InputExample(texts=[text_one, text_two], label=self.cos_sim_matrix[id_one][id_two]))
            if self.max_pairs != -1 and len(self.pos_pairs) > self.max_pairs:
                break
