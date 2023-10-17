from typing import Generator, Iterable, Iterator, List, Optional

import numpy as np
from sentence_transformers import InputExample
from torch.utils.data import IterableDataset

from . import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def sentence_pairs_generation_cos_sim(sentences, pairs, cos_sim_matrix):
    # initialize two empty lists to hold the (sentence, sentence) pairs and
    # labels to indicate if a pair is positive or negative

    idx = list(range(len(sentences)))

    for first_idx in range(len(sentences)):
        current_sentence = sentences[first_idx]
        second_idx = int(np.random.choice([x for x in idx if x != first_idx]))

        cos_sim = float(cos_sim_matrix[first_idx][second_idx])
        paired_sentence = sentences[second_idx]
        pairs.append(InputExample(texts=[current_sentence, paired_sentence], label=cos_sim))

        third_idx = np.random.choice([x for x in idx if x != first_idx])
        cos_sim = float(cos_sim_matrix[first_idx][third_idx])
        paired_sentence = sentences[third_idx]
        pairs.append(InputExample(texts=[current_sentence, paired_sentence], label=cos_sim))

    return pairs


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


class ConstrastiveDataset(IterableDataset):
    def __init__(
        self,
        examples: InputExample,
        multilabel: bool,
        num_iterations: Optional[None] = None,
        sampling_strategy: str = "oversampling",
    ) -> None:
        """Generates positive and negative text pairs for contrastive learning.

        Args:
            examples (InputExample): text and labels in a text transformer dataclass
            multilabel: set to process "multilabel" labels array
            sampling_strategy: "unique", "oversampling", or "undersampling"
            num_iterations: if provided explicitly sets the number of pairs to be generated
                where n_pairs = n_iterations * n_sentences * 2 (for pos & neg pairs)
        """
        super().__init__()
        self.pos_index = 0
        self.neg_index = 0
        self.pos_pairs = []
        self.neg_pairs = []
        self.sentences = np.array([s.texts[0] for s in examples])
        self.labels = np.array([s.label for s in examples])
        self.sentence_labels = list(zip(self.sentences, self.labels))

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

    def generate_multilabel_pairs(self) -> None:
        for (_text, _label), (text, label) in shuffle_combinations(self.sentence_labels):
            if any(np.logical_and(_label, label)):
                # logical_and checks if labels are both set for each class
                self.pos_pairs.append(InputExample(texts=[_text, text], label=1.0))
            else:
                self.neg_pairs.append(InputExample(texts=[_text, text], label=0.0))

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

    def __iter__(self) -> Iterator[InputExample]:
        for pos_pair, neg_pair in zip(self.get_positive_pairs(), self.get_negative_pairs()):
            yield pos_pair
            yield neg_pair

    def __len__(self) -> int:
        return self.len_pos_pairs + self.len_neg_pairs
