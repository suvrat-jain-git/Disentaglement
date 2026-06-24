import torch
import torch.nn as nn

class IdentityHead(nn.Module):
    def __init__(self, node_dim=512, proj_dim=256, hidden_dim=512, num_classes=None):
        super().__init__()
        assert num_classes is not None, (
            "num_classes must be specified — it equals the number of "
            "unique identities in the training split."
        )

        self.fm_proj = nn.Linear(node_dim, proj_dim)
        self.fk_proj = nn.Linear(node_dim, proj_dim)

        fused_dim = proj_dim * 2
        
        self.fc = nn.Linear(fused_dim, hidden_dim, bias=False)

        self.bnneck = nn.BatchNorm1d(hidden_dim)

        self.classifier = nn.Linear(hidden_dim, num_classes, bias=False)

    def _embed(self, Fm_prime, Fk_prime):
        fm_proj = self.fm_proj(Fm_prime)
        fk_proj = self.fk_proj(Fk_prime)

        fused = torch.cat([fm_proj, fk_proj], dim=1)

        embedding = self.fc(fused)

        return embedding

    def forward(self, Fm_prime, Fk_prime):
        embedding = self._embed(Fm_prime, Fk_prime)

        bn_feat = self.bnneck(embedding)

        logits = self.classifier(bn_feat)

        return embedding, logits

    def get_embedding(self, Fm_prime, Fk_prime):
        return self._embed(Fm_prime, Fk_prime)
