from collections import defaultdict
from typing import Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import IterableDataset

from . import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


class ContrastiveDataset(IterableDataset):
    def __init__(
        self,
        sentences: List[str],
        labels: List[Union[int, float]],
        multilabel: bool,
        num_iterations: Optional[int] = None,
        sampling_strategy: str = "oversampling",
        max_pairs: int = -1,
        seed: int = 42,
    ) -> None:
        """Generates positive and negative text pairs for contrastive learning.

        Uses streaming pair generation to avoid O(n²) memory consumption.

        Args:
            sentences (List[str]): text sentences to generate pairs from
            labels (List[Union[int, float]]): labels for each sentence
            multilabel: set to process "multilabel" labels array
            sampling_strategy: "unique", "oversampling", or "undersampling"
            num_iterations: if provided explicitly sets the number of pairs to be generated
                where n_pairs = n_iterations * n_sentences * 2 (for pos & neg pairs)
            max_pairs: If not -1, then we only sample pairs until we have certainly reached
                max_pairs pairs.
            seed: Random seed for reproducibility.
        """
        super().__init__()
        self.sentences = sentences
        self.labels = labels
        self.multilabel = multilabel
        self.sampling_strategy = sampling_strategy
        self.seed = seed
        self.num_samples = len(sentences)

        # Group sample indices by label - O(n) memory
        if multilabel:
            # For multilabel, group by each individual label
            self.label_to_indices = defaultdict(list)
            for idx, label_set in enumerate(labels):
                for lbl in range(len(label_set)):
                    if label_set[lbl]:
                        self.label_to_indices[lbl].append(idx)
            self.unique_labels = list(self.label_to_indices.keys())
        else:
            self.label_to_indices = defaultdict(list)
            for idx, label in enumerate(labels):
                self.label_to_indices[label].append(idx)
            self.unique_labels = list(self.label_to_indices.keys())

        # Calculate possible pairs
        self._calc_possible_pairs()

        # Calculate target pair counts based on strategy
        self._calc_target_pairs(num_iterations, max_pairs)

    def _calc_possible_pairs(self) -> None:
        """Calculate the number of possible positive and negative pairs."""
        # Positive pairs: pairs within same label group
        self.max_pos_pairs = 0
        for label, indices in self.label_to_indices.items():
            n = len(indices)
            self.max_pos_pairs += n * (n - 1) // 2

        # Negative pairs: pairs across different label groups
        # Total pairs - positive pairs
        total_pairs = self.num_samples * (self.num_samples - 1) // 2
        self.max_neg_pairs = total_pairs - self.max_pos_pairs

        # For multilabel, recalculate since samples can belong to multiple groups
        if self.multilabel:
            # Positive = any shared label, Negative = no shared labels
            # This is approximate for multilabel
            pass  # Keep the approximation from single-label calc

    def _calc_target_pairs(self, num_iterations: Optional[int], max_pairs: int) -> None:
        """Calculate target number of pos/neg pairs based on sampling strategy."""
        max_pos_or_neg = -1 if max_pairs == -1 else max_pairs // 2

        if num_iterations is not None and num_iterations > 0:
            self.len_pos_pairs = num_iterations * self.num_samples
            self.len_neg_pairs = num_iterations * self.num_samples
        elif self.sampling_strategy == "unique":
            self.len_pos_pairs = self.max_pos_pairs
            self.len_neg_pairs = self.max_neg_pairs
        elif self.sampling_strategy == "undersampling":
            min_pairs = min(self.max_pos_pairs, self.max_neg_pairs)
            self.len_pos_pairs = min_pairs
            self.len_neg_pairs = min_pairs
        elif self.sampling_strategy == "oversampling":
            max_pairs_count = max(self.max_pos_pairs, self.max_neg_pairs)
            self.len_pos_pairs = max_pairs_count
            self.len_neg_pairs = max_pairs_count
        else:
            raise ValueError(
                "Invalid sampling strategy. Must be one of 'unique', 'oversampling', or 'undersampling'."
            )

        # Apply max_pairs limit
        if max_pos_or_neg != -1:
            self.len_pos_pairs = min(self.len_pos_pairs, max_pos_or_neg)
            self.len_neg_pairs = min(self.len_neg_pairs, max_pos_or_neg)

    @property
    def estimated_num_pairs(self) -> int:
        """Estimated total number of pairs that will be generated."""
        return self.len_pos_pairs + self.len_neg_pairs

    def _sample_positive_pair(
        self, rng: np.random.RandomState, seen_pairs: set
    ) -> Optional[Dict[str, Union[str, float]]]:
        """Sample a positive pair (same label) that hasn't been seen."""
        # Try a limited number of times to find an unseen pair
        for _ in range(100):
            # Pick a random label that has at least 2 samples
            valid_labels = [l for l in self.unique_labels if len(self.label_to_indices[l]) >= 2]
            if not valid_labels:
                return None

            label = valid_labels[rng.randint(len(valid_labels))]
            indices = self.label_to_indices[label]

            # Pick two different indices
            idx1, idx2 = rng.choice(len(indices), size=2, replace=False)
            i, j = indices[idx1], indices[idx2]

            # Normalize pair order for deduplication
            pair_key = (min(i, j), max(i, j), 1)  # 1 = positive

            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                return {
                    "sentence_1": self.sentences[i],
                    "sentence_2": self.sentences[j],
                    "label": 1.0,
                }

        # If we couldn't find a new pair, allow duplicate
        return {
            "sentence_1": self.sentences[i],
            "sentence_2": self.sentences[j],
            "label": 1.0,
        }

    def _sample_negative_pair(
        self, rng: np.random.RandomState, seen_pairs: set
    ) -> Optional[Dict[str, Union[str, float]]]:
        """Sample a negative pair (different labels) that hasn't been seen."""
        if len(self.unique_labels) < 2:
            return None

        for _ in range(100):
            # Pick two different labels
            label1, label2 = rng.choice(self.unique_labels, size=2, replace=False)

            # Pick one index from each
            i = self.label_to_indices[label1][rng.randint(len(self.label_to_indices[label1]))]
            j = self.label_to_indices[label2][rng.randint(len(self.label_to_indices[label2]))]

            # Normalize pair order for deduplication
            pair_key = (min(i, j), max(i, j), 0)  # 0 = negative

            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                return {
                    "sentence_1": self.sentences[i],
                    "sentence_2": self.sentences[j],
                    "label": 0.0,
                }

        # If we couldn't find a new pair, allow duplicate
        return {
            "sentence_1": self.sentences[i],
            "sentence_2": self.sentences[j],
            "label": 0.0,
        }

    def _sample_multilabel_positive_pair(
        self, rng: np.random.RandomState, seen_pairs: set
    ) -> Optional[Dict[str, Union[str, float]]]:
        """Sample a positive pair for multilabel (shares at least one label)."""
        for _ in range(100):
            # Pick a random label that has at least 2 samples
            valid_labels = [l for l in self.unique_labels if len(self.label_to_indices[l]) >= 2]
            if not valid_labels:
                return None

            label = valid_labels[rng.randint(len(valid_labels))]
            indices = self.label_to_indices[label]

            # Pick two different indices (they share at least this label)
            idx1, idx2 = rng.choice(len(indices), size=2, replace=False)
            i, j = indices[idx1], indices[idx2]

            pair_key = (min(i, j), max(i, j), 1)

            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                return {
                    "sentence_1": self.sentences[i],
                    "sentence_2": self.sentences[j],
                    "label": 1.0,
                }

        return {
            "sentence_1": self.sentences[i],
            "sentence_2": self.sentences[j],
            "label": 1.0,
        }

    def _sample_multilabel_negative_pair(
        self, rng: np.random.RandomState, seen_pairs: set
    ) -> Optional[Dict[str, Union[str, float]]]:
        """Sample a negative pair for multilabel (no shared labels)."""
        for _ in range(100):
            # Pick two random samples
            i, j = rng.choice(self.num_samples, size=2, replace=False)

            # Check they don't share any labels
            labels_i = set(idx for idx, val in enumerate(self.labels[i]) if val)
            labels_j = set(idx for idx, val in enumerate(self.labels[j]) if val)

            if not labels_i.intersection(labels_j):
                pair_key = (min(i, j), max(i, j), 0)

                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    return {
                        "sentence_1": self.sentences[i],
                        "sentence_2": self.sentences[j],
                        "label": 0.0,
                    }

        # Fallback - return last attempted pair even if duplicate
        return {
            "sentence_1": self.sentences[i],
            "sentence_2": self.sentences[j],
            "label": 0.0,
        }

    def __iter__(self) -> Generator[Dict[str, Union[str, float]], None, None]:
        """Yield pairs on-the-fly without storing them all in memory."""
        rng = np.random.RandomState(seed=self.seed)
        seen_pairs = set()

        pos_yielded = 0
        neg_yielded = 0

        # Interleave positive and negative pairs
        while pos_yielded < self.len_pos_pairs or neg_yielded < self.len_neg_pairs:
            # Yield a positive pair if needed
            if pos_yielded < self.len_pos_pairs:
                if self.multilabel:
                    pair = self._sample_multilabel_positive_pair(rng, seen_pairs)
                else:
                    pair = self._sample_positive_pair(rng, seen_pairs)
                if pair:
                    yield pair
                    pos_yielded += 1

            # Yield a negative pair if needed
            if neg_yielded < self.len_neg_pairs:
                if self.multilabel:
                    pair = self._sample_multilabel_negative_pair(rng, seen_pairs)
                else:
                    pair = self._sample_negative_pair(rng, seen_pairs)
                if pair:
                    yield pair
                    neg_yielded += 1

    def __len__(self) -> int:
        """Return estimated number of pairs (for compatibility)."""
        return self.len_pos_pairs + self.len_neg_pairs


class ContrastiveDistillationDataset(IterableDataset):
    def __init__(
        self,
        sentences: List[str],
        cos_sim_matrix: torch.Tensor,
        num_iterations: Optional[int] = None,
        sampling_strategy: str = "oversampling",
        max_pairs: int = -1,
        seed: int = 42,
    ) -> None:
        """Generates text pairs with cosine similarity labels for distillation.

        Uses streaming pair generation to avoid O(n²) memory consumption.

        Args:
            sentences (List[str]): text sentences to generate pairs from
            cos_sim_matrix (torch.Tensor): precomputed cosine similarity matrix
            num_iterations: if provided explicitly sets the number of pairs
            sampling_strategy: "unique", "oversampling", or "undersampling"
            max_pairs: If not -1, limits the number of pairs generated
            seed: Random seed for reproducibility.
        """
        super().__init__()
        self.sentences = sentences
        self.cos_sim_matrix = cos_sim_matrix
        self.num_samples = len(sentences)
        self.seed = seed

        # Total possible pairs
        self.max_pairs = self.num_samples * (self.num_samples - 1) // 2

        # Calculate target pairs
        max_pos_or_neg = -1 if max_pairs == -1 else max_pairs

        if num_iterations is not None and num_iterations > 0:
            self.len_pairs = num_iterations * self.num_samples
        else:
            self.len_pairs = self.max_pairs

        if max_pos_or_neg != -1:
            self.len_pairs = min(self.len_pairs, max_pos_or_neg)

        # For compatibility with parent class interface
        self.len_pos_pairs = self.len_pairs
        self.len_neg_pairs = 0

    @property
    def estimated_num_pairs(self) -> int:
        """Estimated total number of pairs that will be generated."""
        return self.len_pairs

    def __iter__(self) -> Generator[Dict[str, Union[str, float]], None, None]:
        """Yield pairs on-the-fly without storing them all in memory."""
        rng = np.random.RandomState(seed=self.seed)
        seen_pairs = set()

        yielded = 0
        while yielded < self.len_pairs:
            # Try to find an unseen pair
            for _ in range(100):
                i, j = rng.choice(self.num_samples, size=2, replace=False)
                pair_key = (min(i, j), max(i, j))

                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    break

            # Yield the pair (even if duplicate after 100 tries)
            yield {
                "sentence_1": self.sentences[i],
                "sentence_2": self.sentences[j],
                "label": float(self.cos_sim_matrix[i][j]),
            }
            yielded += 1

    def __len__(self) -> int:
        """Return estimated number of pairs (for compatibility)."""
        return self.len_pairs
