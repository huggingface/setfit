import copy
import os
from dataclasses import dataclass

import joblib
import numpy as np
import requests
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
from sentence_transformers import InputExample, SentenceTransformer, models
from sklearn.linear_model import LogisticRegression

from . import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


MODEL_HEAD_NAME = "model_head.pkl"


class SetFitBaseModel:
    def __init__(self, model, max_seq_length: int, add_normalization_layer: bool) -> None:
        self.model = SentenceTransformer(model)
        self.model_original_state = copy.deepcopy(self.model.state_dict())
        self.model.max_seq_length = max_seq_length

        if add_normalization_layer:
            self.model._modules["2"] = models.Normalize()


@dataclass
class SetFitModel(PyTorchModelHubMixin):
    """A SetFit model with integration to the Hugging Face Hub."""

    def __init__(self, model_body=None, model_head=None):
        super(SetFitModel, self).__init__()
        self.model_body = model_body
        self.model_head = model_head

        self.model_original_state = copy.deepcopy(self.model_body.state_dict())

    def fit(self, x_train, y_train):
        embeddings = self.model_body.encode(x_train)
        self.model_head.fit(embeddings, y_train)

    def predict(self, x_test):
        embeddings = self.model_body.encode(x_test)
        return self.model_head.predict(embeddings)

    def predict_proba(self, x_test):
        embeddings = self.model_body.encode(x_test)
        return self.model_head.predict_proba(embeddings)

    def __call__(self, inputs):
        embeddings = self.model_body.encode(inputs)
        return self.model_head.predict(embeddings)

    def _save_pretrained(self, save_directory):
        self.model_body.save(path=save_directory)
        joblib.dump(self.model_head, f"{save_directory}/{MODEL_HEAD_NAME}")

    @classmethod
    def _from_pretrained(
        cls,
        model_id: str,
        revision=None,
        cache_dir=None,
        force_download=None,
        proxies=None,
        resume_download=None,
        local_files_only=None,
        use_auth_token=None,
        **model_kwargs,
    ):
        model_body = SentenceTransformer(model_id)

        if os.path.isdir(model_id) and MODEL_HEAD_NAME in os.listdir(model_id):
            model_head_file = os.path.join(model_id, MODEL_HEAD_NAME)
        else:
            try:
                model_head_file = hf_hub_download(
                    repo_id=model_id,
                    filename=MODEL_HEAD_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    use_auth_token=use_auth_token,
                    local_files_only=local_files_only,
                )
            except requests.exceptions.RequestException:
                logger.info(
                    f"{MODEL_HEAD_NAME} not found on HuggingFace Hub, initialising classification head with random weights."
                    " You should TRAIN this model on a downstream task to use it for predictions and inference."
                )
                model_head_file = None

        if model_head_file is not None:
            model_head = joblib.load(model_head_file)
        else:
            model_head = LogisticRegression()
        return SetFitModel(model_body=model_body, model_head=model_head)


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.

    It also supports the unsupervised contrastive loss in SimCLR.
    """

    def __init__(self, model, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.model = model
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, sentence_features, labels=None, mask=None):
        """Computes loss for model.

        If both `labels` and `mask` are None, it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.

        Returns:
            A loss scalar.
        """
        features = self.model(sentence_features[0])["sentence_embedding"]

        # Normalize embeddings
        features = torch.nn.functional.normalize(features, p=2, dim=1)

        # Add n_views dimension
        features = torch.unsqueeze(features, 1)

        device = features.device

        if len(features.shape) < 3:
            raise ValueError("`features` needs to be [bsz, n_views, ...]," "at least 3 dimensions are required")
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # Compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # Mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1, torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask

        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


def sentence_pairs_generation(sentences, labels, pairs):
    # Initialize two empty lists to hold the (sentence, sentence) pairs and
    # labels to indicate if a pair is positive or negative

    num_classes = np.unique(labels)
    idx = [np.where(labels == i)[0] for i in num_classes]

    for first_idx in range(len(sentences)):
        current_sentence = sentences[first_idx]
        label = labels[first_idx]
        second_idx = np.random.choice(idx[np.where(num_classes == label)[0][0]])
        positive_sentence = sentences[second_idx]
        # Prepare a positive pair and update the sentences and labels
        # lists, respectively
        pairs.append(InputExample(texts=[current_sentence, positive_sentence], label=1.0))

        negative_idx = np.where(labels != label)[0]
        negative_sentence = sentences[np.random.choice(negative_idx)]
        # Prepare a negative pair of images and update our lists
        pairs.append(InputExample(texts=[current_sentence, negative_sentence], label=0.0))
    # Return a 2-tuple of our image pairs and labels
    return pairs


def sentence_pairs_generation_multilabel(sentences, labels, pairs):
    # Initialize two empty lists to hold the (sentence, sentence) pairs and
    # labels to indicate if a pair is positive or negative
    for first_idx in range(len(sentences)):
        current_sentence = sentences[first_idx]
        sample_labels = np.where(labels[first_idx, :] == 1)[0]
        if len(np.where(labels.dot(labels[first_idx, :].T) == 0)[0]) == 0:
            continue
        else:

            for _label in sample_labels:
                second_idx = np.random.choice(np.where(labels[:, _label] == 1)[0])
                positive_sentence = sentences[second_idx]
                # Prepare a positive pair and update the sentences and labels
                # lists, respectively
                pairs.append(InputExample(texts=[current_sentence, positive_sentence], label=1.0))

            # Search for sample that don't have a label in common with current
            # sentence
            negative_idx = np.where(labels.dot(labels[first_idx, :].T) == 0)[0]
            negative_sentence = sentences[np.random.choice(negative_idx)]
            # Prepare a negative pair of images and update our lists
            pairs.append(InputExample(texts=[current_sentence, negative_sentence], label=0.0))
    # Return a 2-tuple of our image pairs and labels
    return pairs


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


class SKLearnWrapper:
    def __init__(self, st_model=None, clf=None):
        self.st_model = st_model
        self.clf = clf

    def fit(self, x_train, y_train):
        embeddings = self.st_model.encode(x_train)
        self.clf.fit(embeddings, y_train)

    def predict(self, x_test):
        embeddings = self.st_model.encode(x_test)
        return self.clf.predict(embeddings)

    def predict_proba(self, x_test):
        embeddings = self.st_model.encode(x_test)
        return self.clf.predict_proba(embeddings)

    def save(self, path):
        self.st_model.save(path=path)
        joblib.dump(self.clf, f"{path}/setfit_head.pkl")

    def load(self, path):
        self.st_model = SentenceTransformer(model_name_or_path=path)
        self.clf = joblib.load(f"{path}/setfit_head.pkl")
