from itertools import zip_longest
from typing import Generator, Iterable, List

import numpy as np
from sentence_transformers import InputExample
from torch.utils.data import IterableDataset

from . import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


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


class ConstrastiveDataset(IterableDataset):
    def __init__(self, examples, num_iterations, unique_pairs, multilabel):
        """Generates positive and negative sentence pairs for contrastive learning.

        Args:
            examples (InputExample): text and labels in a sentence transformer dataclass
            num_iterations: sets the number of contastive sample pairs to be generated
            unique_pairs: when true will only return upto the number of unique sentence
                pair combinations avaliable
            multilabel: set to process "multilabel" labels array
        """
        super().__init__()

        self.pos_index = 0
        self.neg_index = 0
        self.multilabel = multilabel
        self.unique_pairs = unique_pairs
        self.sentences = np.array([s.texts[0] for s in examples])
        self.labels = np.array([s.label for s in examples])
        self.max_pairs = num_iterations * len(examples)

        # generate dataset so __len__ method can be used
        self.generate_sentence_pairs()

    def generate_sentence_pairs(self) -> None:
        """Generates a new batch of positive and negative sentence pairs.

        Note: pos_index/ neg_index keep the position of pairs being generated.
        """
        positive_pairs = self.positive_sentence_pairs(self.max_pairs, self.unique_pairs)
        negative_pairs = self.negative_sentence_pairs(self.max_pairs, self.unique_pairs)

        if self.unique_pairs:
            extra_pairs = abs(len(positive_pairs) - len(negative_pairs))

            if len(positive_pairs) > len(negative_pairs):
                logger.warning(
                    f"** Oversampling ({extra_pairs:,}) negative pairs to balance contrastive training samples."
                )
                negative_pairs += self.negative_sentence_pairs(max_pairs=extra_pairs, unique_pairs=False)

            if len(negative_pairs) > len(positive_pairs):
                logger.warning(
                    f"** Oversampling ({extra_pairs:,}) positive pairs to balance contrastive training samples."
                )
                positive_pairs += self.positive_sentence_pairs(max_pairs=extra_pairs, unique_pairs=False)

        self._num_pairs = len(positive_pairs) + len(negative_pairs)
        self.positive_pairs = positive_pairs
        self.negative_pairs = negative_pairs

    def positive_sentence_pairs(self, max_pairs: int, unique_pairs: bool = False) -> List[InputExample]:
        """Generates all unique or upto a max no. of combinations of positive sentence pairs.

        Samples positive combinations of sentences (without replacement) and maximises
            sampling of different classes in the pairs being generated.

        Args:
            max_pairs: returns when this many pairs are generated
            unique_pairs: if true will return sentences if all unique combinations,
                before max_pairs count is reached

        Returns:
            List of positive sentence pairs (upto the no. of unique_pairs or max_pairs)
        """
        labels = self.labels
        sentences = self.sentences
        multilabel = self.multilabel
        pairs = []

        if multilabel:
            label_ids = np.arange(labels.shape[1])  # based on index = 1,0
        else:
            label_ids = np.unique(labels)  # based on class int
        while True:
            index = 0
            positive_combinators = []
            for _label in label_ids:
                if multilabel:
                    label_sentences = sentences[np.where(labels[:, _label] == 1)[0]]
                else:
                    label_sentences = sentences[np.where(labels == _label)]
                positive_combinators.append(shuffle_combinations(label_sentences, replacement=True))

            for pos_pairs in zip_longest(*positive_combinators):
                for pos_pair in pos_pairs:
                    if pos_pair is not None:
                        index += 1
                        if index > self.pos_index:
                            pairs.append(InputExample(texts=[*pos_pair], label=1.0))
                            if len(pairs) == max_pairs:
                                self.pos_index = index
                                return pairs
            self.pos_index = 0
            if unique_pairs:
                break
        logger.warning(f"** All ({len(pairs):,}) positive unique pairs generated")
        return pairs

    def negative_sentence_pairs(self, max_pairs: int, unique_pairs: bool = False) -> List[InputExample]:
        """Generates all or upto a max sample no. of negative combinations.

        Randomly samples negative combinations of sentences (without replacement).

        Args:
            max_pairs: returns when this many pairs are generated
            unique_pairs: if true will return sentences if all unique combinations,
                before max_pairs count is reached

        Returns:
            List of negative sentence pairs (upto the no. of unique_pairs or max_pairs)
        """
        multilabel = self.multilabel
        pairs = []

        sentence_labels = list(zip(self.sentences, self.labels))
        while True:
            index = 0
            for (_sentence, _label), (sentence, label) in shuffle_combinations(sentence_labels):
                # logical_and checks if labels are both set for each class
                if (multilabel and not any(np.logical_and(_label, label))) or (not multilabel and _label != label):
                    index += 1
                    if index > self.neg_index:
                        pairs.append(InputExample(texts=[_sentence, sentence], label=0.0))
                        if len(pairs) == max_pairs:
                            self.neg_index = index
                            return pairs
            self.neg_index = 0
            if unique_pairs:
                break
        logger.warning(f"** All ({len(pairs):,}) negative unique pairs generated")
        return pairs

    def __iter__(self):
        for pos_pair, neg_pair in zip(self.positive_pairs, self.negative_pairs):
            # generates one of each in turn
            yield pos_pair
            yield neg_pair

        if self.pos_index or self.neg_index:
            # not all pairs combinations sampled so continues from last index
            self.generate_sentence_pairs()

    def __len__(self):
        return self._num_pairs
