import torch
import torch.nn as nn


class IdentityHead(nn.Module):
    """
    Fused identity head with projection, BNNeck, and dual-loss outputs.
    """

    def __init__(self, node_dim=512, proj_dim=256, hidden_dim=512, num_classes=None):
        """
        Args:
            node_dim:    input dim of Fm' and Fk' (both 512)
            proj_dim:    output dim of each projection layer (256)
            hidden_dim:  FC layer size after concat (512 = proj_dim * 2)
            num_classes: number of training identities — MUST be set at init
        """
        super().__init__()
        assert num_classes is not None, (
            "num_classes must be specified — it equals the number of "
            "unique identities in the training split."
        )

        # Projection layers — one per branch.
        # Each compresses 512 → 256, forcing a compact summary.
        # bias=True: no BN after projection, so bias is meaningful.
        self.fm_proj = nn.Linear(node_dim, proj_dim)
        self.fk_proj = nn.Linear(node_dim, proj_dim)

        # After concat: [B, proj_dim*2] = [B, 512]
        # FC maps this to the embedding space.
        # bias=False: BNNeck immediately follows, which has its own
        # learnable shift (beta) — conv/linear bias would be redundant.
        fused_dim = proj_dim * 2  # 256 + 256 = 512
        self.fc = nn.Linear(fused_dim, hidden_dim, bias=False)

        # BNNeck: a single BatchNorm1d layer with no affine transformation
        # disabled... actually we KEEP affine=True (default) here.
        # The BN learns to re-scale and re-center the embedding for the
        # classifier. The key is that triplet loss sees the pre-BN embedding
        # while CE loss sees the post-BN feature.
        self.bnneck = nn.BatchNorm1d(hidden_dim)

        # Classifier: maps BN-normalised feature to identity logits.
        # bias=False: BNNeck already provides a learned shift via beta.
        self.classifier = nn.Linear(hidden_dim, num_classes, bias=False)

    def _embed(self, Fm_prime, Fk_prime):
        """
        Internal helper: runs the shared projection + FC path.
        Returns the pre-BNNeck embedding [B, 512].
        Both forward() and get_embedding() call this — no code duplication.
        """
        # Project each branch independently
        # [B, 512] -> [B, 256]
        fm_proj = self.fm_proj(Fm_prime)
        fk_proj = self.fk_proj(Fk_prime)

        # Concatenate along feature dimension
        # [B, 256] + [B, 256] -> [B, 512]
        fused = torch.cat([fm_proj, fk_proj], dim=1)

        # FC: [B, 512] -> [B, 512]
        # This is the identity embedding — the representation we care about
        # for retrieval, t-SNE, and triplet loss.
        embedding = self.fc(fused)

        return embedding

    def forward(self, Fm_prime, Fk_prime):
        """
        Args:
            Fm_prime: [B, 512] — morphology after graph
            Fk_prime: [B, 512] — motion after graph

        Returns:
            embedding: [B, 512] — pre-BNNeck, use for triplet loss and retrieval
            logits:    [B, num_classes] — post-BNNeck, use for CE loss
        """
        # Shared path: projection + concat + FC
        embedding = self._embed(Fm_prime, Fk_prime)

        # BNNeck: normalise for classifier, preserve metric geometry for triplet.
        bn_feat = self.bnneck(embedding)

        # Classifier: [B, 512] -> [B, num_classes]
        logits = self.classifier(bn_feat)

        return embedding, logits

    def get_embedding(self, Fm_prime, Fk_prime):
        """
        Returns the pre-BNNeck embedding only.
        Use at test time for nearest-neighbour retrieval and analysis.

        At test time we do NOT want BNNeck normalization — we want the
        raw metric-space embedding that triplet loss shaped during training.

        Args:
            Fm_prime: [B, 512]
            Fk_prime: [B, 512]

        Returns:
            embedding: [B, 512]
        """
        return self._embed(Fm_prime, Fk_prime)
