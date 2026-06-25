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

        # Project Fk only: 512 → 256
        # Identity head uses Fk exclusively — Fm is not used here.
        # Fm is supervised by the gender head only.
        self.fm_proj = nn.Linear(node_dim, proj_dim)
        self.fk_proj = nn.Linear(node_dim, proj_dim)

        # bias=False: BNNeck immediately follows.
        # FC maps concat(Fm, Fk) → embedding
        fused_dim = proj_dim * 2
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
        Internal helper: returns the pre-BNNeck embedding [B, 512].

        Identity uses Fk_prime ONLY.
        Fm_prime argument is accepted for API compatibility but not used here.

        Rationale:
            If identity supervision flows through concat(Fm, Fk), identity
            gradients push gender information into Fm, contaminating the
            morphology branch. Using Fk only means:
                Fm is shaped purely by gender supervision  (body shape)
                Fk is shaped by identity + triplet          (gait dynamics)
            Fm still contributes context to Fk via the graph interaction.
        """
        # Project Fk_prime: [B, 512] -> [B, 256]
        fm_proj = self.fm_proj(Fm_prime)
        fk_proj = self.fk_proj(Fk_prime)

        # FC: [B, 256] -> [B, 512] — identity embedding
        fused = torch.cat([fm_proj, fk_proj], dim=1)
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
