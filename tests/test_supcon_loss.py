import torch
from torch import nn

from setfit.losses import SupConLoss


class _Embeddings(nn.Module):
    def forward(self, features):
        return {"sentence_embedding": features}


def test_a_class_that_appears_once_does_not_make_the_loss_nan():
    loss_fn = SupConLoss(_Embeddings())
    features = torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]])
    labels = torch.tensor([0, 0, 1])

    loss = loss_fn([features], labels)
    assert torch.isfinite(loss)

    paired = loss_fn([features[:2]], labels[:2])
    assert torch.isfinite(paired)
    assert torch.allclose(paired, torch.tensor(1.4901161e-08), atol=1e-6)

    alone = loss_fn([features[1:]], torch.tensor([0, 1]))
    assert torch.isfinite(alone)
    assert alone.item() == 0.0
