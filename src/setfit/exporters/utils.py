import torch


def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Perform attention-aware mean pooling.

    This method takes in embeddings of shape (batch, sequence, embedding_size) and performs average
    pooling across the sequence dimension to yield embeddings of size (batch, embedding_size).

    From:
    https://github.com/UKPLab/sentence-transformers/blob/0b5ef4be93d2b21de3a918a084b48aab6ba48595/sentence_transformers/model_card_templates.py#L134  # noqa: E501

    Args:
        token_embeddings (`torch.Tensor`): The embeddings we wish to pool over of shape
            (batch, sequence, embedding_size).  This will pool over the sequence to yield
            (batch, embedding_size).
        attention_mask (`torch.Tensor`): The binary attention mask across the embedings of shape

    Returns:
        (`torch.Tensor`) The mean pooled embeddings of size (batch, embedding_size).
    """
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
