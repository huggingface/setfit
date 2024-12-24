from itertools import combinations, zip_longest
from typing import Dict, Generator, Iterable, List, Literal, Optional, Union
from collections import Counter

import numpy as np
import torch
from torch.utils.data import IterableDataset
from transformers.utils import ExplicitEnum

from . import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


class SamplingStrategy(ExplicitEnum):
    """
    ## Oversampling

    By default, SetFit applies the oversampling strategy for its contrastive pairs. This strategy samples an equal amount of positive and negative training
    pairs, oversampling the minority pair type to match that of the majority pair type. As the number of negative pairs is generally larger than the number
    of positive pairs, this usually involves oversampling the positive pairs.

    In our running example, this would involve oversampling the 62 positive pairs up to 128, resulting in one epoch of 128 + 128 = 256 pairs. In summary:

    * Y An equal amount of positive and negative pairs are sampled.
    * Y Every possible pair is used.
    * X There is some data duplication.

    ## Undersampling

    Like oversampling, this strategy samples an equal amount of positive and negative training pairs. However, it undersamples the majority pair type to match
    that of the minority pair type. This usually involves undersampling the negative pairs to match the positive pairs.

    In our running example, this would involve undersampling the 128 negative pairs down to 62, resulting in one epoch of 62 + 62 = 124 pairs. In summary:

    * Y An equal amount of positive and negative pairs are sampled.
    * X **Not** every possible pair is used.
    * Y There is **no** data duplication.

    ## Unique

    Thirdly, the unique strategy does not sample an equal amount of positive and negative training pairs. Instead, it simply samples all possible pairs exactly
     once. No form of oversampling or undersampling is used here.

    In our running example, this would involve sampling all negative and positive pairs, resulting in one epoch of 62 + 128 = 190 pairs. In summary:

    * X **Not** an equal amount of positive and negative pairs are sampled.
    * Y Every possible pair is used.
    * Y There is **no** data duplication.
    """
    OVERSAMPLING = "oversampling"
    UNDERSAMPLING = "undersampling"
    UNIQUE = "unique"


def shuffle_combinations(iterable: Iterable, replacement: bool = False) -> Generator:
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


class ContrastiveDatasetIt(IterableDataset):
    def __init__(
        self,
        sentences: List[str],
        labels: List[Union[int, float]],
        multilabel: bool = False,  # False for now
        num_iterations: Optional[None] = None,
        sampling_strategy: Literal["oversampling", "undersampling", "unique"] = "oversampling",
        max_pairs: int = -1,
    ) -> None:
        """Generates positive and negative text pairs for contrastive learning.

        Args:
            sentences (List[str]): text sentences to generate pairs from
            labels (List[Union[int, float]]): labels for each sentence
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
        self.sentences = sentences
        self.labels = labels
        self.sentence_labels = list(zip(self.sentences, self.labels))
        self.max_pos_or_neg = np.inf if max_pairs == -1 else max_pairs // 2
        self._multilabel = multilabel

        sampling_strategy = SamplingStrategy(sampling_strategy)

        # calculate number of positive and negative combinations
        label_counts = Counter(labels)
        # postive number of pairs from an n element set without replacement
        self.total_pos_pairs = int(sum([n * (n - 1) / 2 for n in label_counts.values()]))
        # negative product
        self.total_neg_pairs = self.total_neg_pairs = sum(a * b for a, b in combinations(label_counts.values(), 2))

        if num_iterations is not None and num_iterations > 0:
            self.len_pos_pairs = num_iterations * len(self.sentences)
            self.len_neg_pairs = num_iterations * len(self.sentences)

        elif sampling_strategy == SamplingStrategy.UNIQUE:
            self.len_pos_pairs = int(np.min([self.total_pos_pairs, self.max_pos_or_neg]))
            self.len_neg_pairs = int(np.min([self.total_neg_pairs, self.max_pos_or_neg]))

        elif sampling_strategy == SamplingStrategy.UNDERSAMPLING:
            self.len_pos_pairs = int(np.min([min(self.total_pos_pairs, self.total_neg_pairs), self.max_pos_or_neg]))
            self.len_neg_pairs = int(np.min([min(self.total_pos_pairs, self.total_neg_pairs), self.max_pos_or_neg]))

        elif sampling_strategy == SamplingStrategy.OVERSAMPLING:
            self.len_pos_pairs = int(np.min([max(self.total_pos_pairs, self.total_neg_pairs), self.max_pos_or_neg]))
            self.len_neg_pairs = int(np.min([max(self.total_pos_pairs, self.total_neg_pairs), self.max_pos_or_neg]))

    # generate pair functions are not ideal but still wont blow the memory if you decide to train on big dataset
    def generate_positive_pair(self):
        pair_generator = shuffle_combinations(self.sentence_labels)
        while True:
            for (_text, _label), (text, label) in pair_generator:
                if self._multilabel:
                    is_positive = any(np.logical_and(_label, label))
                else:
                    is_positive = _label == label

                if is_positive:
                    yield {"sentence_1": _text, "sentence_2": text, "label": 1.0}
            # restart
            pair_generator = shuffle_combinations(self.sentence_labels)

    def generate_negative_pair(self):
        pair_generator = shuffle_combinations(self.sentence_labels)
        while True:
            for (_text, _label), (text, label) in pair_generator:
                if self._multilabel:
                    is_negative = not any(np.logical_and(_label, label))
                else:
                    is_negative = _label != label

                if is_negative:
                    yield {"sentence_1": _text, "sentence_2": text, "label": 0.0}
            pair_generator = shuffle_combinations(self.sentence_labels)

    def __iter__(self):

        generated_pos_pairs = 0
        generated_neg_pairs = 0

        pos_generator = self.generate_positive_pair()
        neg_generator = self.generate_negative_pair()

        while (generated_pos_pairs + generated_neg_pairs) < len(self):
            if generated_pos_pairs < self.len_pos_pairs:
                yield next(pos_generator)
                generated_pos_pairs += 1
            if generated_neg_pairs < self.len_neg_pairs:
                yield next(neg_generator)
                generated_neg_pairs += 1

    def __len__(self) -> int:
        return self.len_pos_pairs + self.len_neg_pairs


class ContrastiveDataset(IterableDataset):
    def __init__(
        self,
        sentences: List[str],
        labels: List[Union[int, float]],
        multilabel: bool,
        num_iterations: Optional[None] = None,
        sampling_strategy: str = "oversampling",
        max_pairs: int = -1,
    ) -> None:
        """Generates positive and negative text pairs for contrastive learning.

        Args:
            sentences (List[str]): text sentences to generate pairs from
            labels (List[Union[int, float]]): labels for each sentence
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
        self.sentences = sentences
        self.labels = labels
        self.sentence_labels = list(zip(self.sentences, self.labels))
        self.max_pos_or_neg = -1 if max_pairs == -1 else max_pairs // 2

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
            is_positive = _label == label
            is_positive_full = self.max_pos_or_neg != -1 and len(self.pos_pairs) >= self.max_pos_or_neg
            is_negative_full = self.max_pos_or_neg != -1 and len(self.neg_pairs) >= self.max_pos_or_neg

            if is_positive:
                if not is_positive_full:
                    self.pos_pairs.append({"sentence_1": _text, "sentence_2": text, "label": 1.0})
            elif not is_negative_full:
                self.neg_pairs.append({"sentence_1": _text, "sentence_2": text, "label": 0.0})

            if is_positive_full and is_negative_full:
                break

    def generate_multilabel_pairs(self) -> None:
        for (_text, _label), (text, label) in shuffle_combinations(self.sentence_labels):
            # logical_and checks if labels are both set for each class
            is_positive = any(np.logical_and(_label, label))
            is_positive_full = self.max_pos_or_neg != -1 and len(self.pos_pairs) >= self.max_pos_or_neg
            is_negative_full = self.max_pos_or_neg != -1 and len(self.neg_pairs) >= self.max_pos_or_neg

            if is_positive:
                if not is_positive_full:
                    self.pos_pairs.append({"sentence_1": _text, "sentence_2": text, "label": 1.0})
            elif not is_negative_full:
                self.neg_pairs.append({"sentence_1": _text, "sentence_2": text, "label": 0.0})

            if is_positive_full and is_negative_full:
                break

    def get_positive_pairs(self) -> List[Dict[str, Union[str, float]]]:
        pairs = []
        for _ in range(self.len_pos_pairs):
            if self.pos_index >= len(self.pos_pairs):
                self.pos_index = 0
            pairs.append(self.pos_pairs[self.pos_index])
            self.pos_index += 1
        return pairs

    def get_negative_pairs(self) -> List[Dict[str, Union[str, float]]]:
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
        sentences: List[str],
        cos_sim_matrix: torch.Tensor,
        num_iterations: Optional[None] = None,
        sampling_strategy: str = "oversampling",
        max_pairs: int = -1,
    ) -> None:
        self.cos_sim_matrix = cos_sim_matrix
        super().__init__(
            sentences,
            [0] * len(sentences),
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
            self.pos_pairs.append(
                {"sentence_1": text_one, "sentence_2": text_two, "label": self.cos_sim_matrix[id_one][id_two]}
            )
            if self.max_pos_or_neg != -1 and len(self.pos_pairs) > self.max_pos_or_neg:
                break
