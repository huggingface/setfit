from typing import Generator, Iterable, List
import numpy as np
from torch.utils.data import  IterableDataset
from itertools import zip_longest

from sentence_transformers import InputExample

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


def positive_sentence_pairs_generate(
    sentences: np.ndarray, labels: np.ndarray, max_pairs: int, unique_pairs: bool = False, multilabel: bool = False
) -> List[InputExample]:
    """Generates all unique or upto a max no. of combinations of positive sentence pairs.

    Samples positive combinations of sentences (without replacement) and maximises
        sampling of different classes in the pairs being generated.

    Args:
        sentences: an array of all sentences
        labels: an array of the label_id for each item in `sentences`
        max_pairs: returns when this many pairs are generated
        unique_pairs: if true will return sentences if all unique combinations,
            before max_pairs count is reached
        multilabel: set to process "multilabel" labels array

    Returns:
        List of positive sentence pairs (upto the no. of unique_pairs or max_pairs)
    """
    pairs = []
    if multilabel:
        label_ids = np.arange(labels.shape[1])  # based on index = 1,0
    else:
        label_ids = np.unique(labels)  # based on class int
    while True:
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
                    pairs.append(InputExample(texts=[*pos_pair], label=1.0))
                    if len(pairs) == max_pairs:
                        return pairs
        if unique_pairs:
            break
    logger.warning(f"** All ({len(pairs):,}) positive unique pairs generated")
    return pairs


def negative_sentence_pairs_generate(
    sentences: np.ndarray,
    labels: np.ndarray,
    max_pairs: int,
    unique_pairs: bool = False,
    multilabel: bool = False,
) -> List[InputExample]:
    """Generates all or upto a max sample no. of negative combinations.

    Randomly samples negative combinations of sentences (without replacement)

    Args:
        sentences: an array of all sentences
        labels: an array of the label_id for each item in `sentences`
        max_pairs: returns when this many pairs are generated
        unique_pairs: if true will return sentences if all unique combinations,
            before max_pairs count is reached
        multilabel: set to process "multilabel" labels array

    Returns:
        List of negative sentence pairs (upto the no. of unique_pairs or max_pairs)
    """
    pairs = []
    sentence_labels = list(zip(sentences, labels))
    while True:
        for (_sentence, _label), (sentence, label) in shuffle_combinations(sentence_labels):
            # logical_and checks if labels are both set for each class
            if (multilabel and not any(np.logical_and(_label, label))) or (not multilabel and _label != label):
                pairs.append(InputExample(texts=[_sentence, sentence], label=0.0))
                if len(pairs) == max_pairs:
                    return pairs
        if unique_pairs:
            break
    logger.warning(f"** All ({len(pairs):,}) negative unique pairs generated")
    return pairs


def sentence_pairs_generation(
    sentences: np.ndarray,
    labels: np.ndarray,
    num_iterations: int,
    unique_pairs: bool = False,
    multilabel: bool = False,
) -> List[InputExample]:
    """Generates positive and negative sentence pairs for contrastive learning.

    Args:
        sentences (ArrayLike str): an array of all sentences
        labels (ArrayLike int): an array of the label_id for each item in `sentences`
        num_iterations: sets the number of contastive sample pairs to be generated
        unique_pairs: when true will only return upto the number of unique sentence
            pair combinations avaliable

    Returns:
        List of sentence pairs
    """
    max_pairs = num_iterations * len(sentences)
    positive_pairs = positive_sentence_pairs_generate(sentences, labels, max_pairs, unique_pairs, multilabel)
    negative_pairs = negative_sentence_pairs_generate(sentences, labels, max_pairs, unique_pairs, multilabel)

    if unique_pairs:
        extra_pairs = abs(len(positive_pairs) - len(negative_pairs))

        if len(positive_pairs) > len(negative_pairs):
            logger.warning("** Oversampling negative pairs to balance contrastive training samples.")
            negative_pairs += negative_sentence_pairs_generate(sentences, labels, extra_pairs, False, multilabel)

        if len(negative_pairs) > len(positive_pairs):
            logger.warning("** Oversampling positive pairs to balance contrastive training samples.")
            positive_pairs += positive_sentence_pairs_generate(sentences, labels, extra_pairs, False, multilabel)

    return positive_pairs + negative_pairs


class ConstrastiveDataset(IterableDataset):
    def __init__(self, x_train, y_train, num_iterations, unique_pairs, multilabel):
        super().__init__()

        self.train_examples = sentence_pairs_generation(
            np.array(x_train), np.array(y_train), num_iterations, unique_pairs, multilabel
        )

    def __iter__(self):
        for example in self.train_examples:
            yield example

    def __len__(self):
        return len(self.train_examples)
