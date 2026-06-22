"""
src/losses.py
-------------
Custom loss functions used during hierarchical training.

  • FocalLoss                  — handles class imbalance by down-weighting
                                 easy examples.
  • HierarchicalContrastiveLoss — pushes same-class features together and
                                  different-class features apart, applied
                                  to both unimodal and pairwise feature
                                  representations.
"""

from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for dense prediction / imbalanced classification.

    Reference
    ---------
    Lin et al., "Focal Loss for Dense Object Detection," ICCV 2017.

    Parameters
    ----------
    alpha : float
        Scaling factor for the focal term.
    gamma : float
        Focusing parameter. gamma=0 recovers cross-entropy.
    reduction : str
        ``'mean'`` or ``'sum'``.
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss     = F.cross_entropy(inputs, targets, reduction="none")
        pt          = torch.exp(-ce_loss)
        focal_loss  = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        return focal_loss.sum()


class HierarchicalContrastiveLoss(nn.Module):
    """
    Contrastive loss operating on unimodal and pairwise feature dictionaries.

    For each feature set (and each pair of feature sets) the loss pulls
    same-label embeddings together and pushes different-label embeddings
    apart in the unit-hypersphere embedding space.

    Parameters
    ----------
    temperature : float
        Softmax temperature τ.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: dict, labels: torch.Tensor) -> torch.Tensor:
        loss = torch.tensor(0.0, device=labels.device)

        # Unimodal contrastive terms
        for feat in features.values():
            loss = loss + self._contrastive(feat, labels)

        # Cross-modal contrastive terms
        feat_list = list(features.values())
        for f1, f2 in combinations(feat_list, 2):
            loss = loss + self._contrastive(f1, labels, f2)

        return loss

    def _contrastive(
        self,
        feat1: torch.Tensor,
        labels: torch.Tensor,
        feat2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        feat1 = F.normalize(feat1, dim=1)
        feat2 = F.normalize(feat2, dim=1) if feat2 is not None else feat1

        N   = labels.size(0)
        sim = torch.matmul(feat1, feat2.T) / self.temperature   # [N, N]

        # Remove self-similarity entries
        mask_diag   = ~torch.eye(N, dtype=torch.bool, device=feat1.device)
        sim_no_diag = sim[mask_diag].view(N, N - 1)

        # Build positive / negative masks (excluding diagonal)
        label_col   = labels.contiguous().view(-1, 1)
        pos_mask    = torch.eq(label_col, labels.T)[mask_diag].view(N, N - 1)

        positives = sim_no_diag[pos_mask]
        negatives = sim_no_diag[~pos_mask]

        if positives.numel() == 0:
            return torch.tensor(0.0, device=feat1.device)

        logits = torch.cat(
            [positives.unsqueeze(1), negatives.view(N, -1)], dim=1
        )
        targets = torch.zeros(logits.size(0), dtype=torch.long, device=feat1.device)
        return F.cross_entropy(logits, targets)
