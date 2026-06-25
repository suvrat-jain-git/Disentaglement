import torch
import torch.nn.functional as F


def orthogonality_loss(Fm, Fk):
    """
    Compute orthogonality loss between morphology and motion features.

    Args:
        Fm: [B, D] — morphology features (pre-graph or post-graph)
        Fk: [B, D] — motion features (pre-graph or post-graph)

    Returns:
        loss: scalar — mean squared dot product between normalised features
    """
    # L2 normalise both feature sets
    # After normalisation, each row is a unit vector on the hypersphere
    Fm_norm = F.normalize(Fm, dim=1)   # [B, D]
    Fk_norm = F.normalize(Fk, dim=1)   # [B, D]

    # Cross-correlation matrix: [B, B]
    # entry (i, j) = dot product between sample i of Fm and sample j of Fk
    # We want all entries to be close to zero
    cross = torch.mm(Fm_norm, Fk_norm.t())   # [B, B]

    # Mean squared Frobenius norm — penalises any alignment
    loss = cross.pow(2).mean()

    return loss
