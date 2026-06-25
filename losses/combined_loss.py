import torch
import torch.nn as nn
import torch.nn.functional as F
from losses.triplet import TripletLoss
from losses.orthogonality import orthogonality_loss


class CombinedLoss:

    def __init__(self, w_identity=0.5, w_triplet=1.0, w_gender=0.5,
                 w_orthogonality=0.05, triplet_margin=0.5,
                 num_classes=None, **kwargs):
        self.w_identity     = w_identity
        self.w_triplet      = w_triplet
        self.w_gender       = w_gender
        self.w_orthogonality= w_orthogonality

        self.ce_identity    = nn.CrossEntropyLoss()
        self.triplet        = TripletLoss(margin=triplet_margin)

        # Class-weighted gender CE — Female gets 2x weight
        self.gender_weights = torch.tensor([1.0, 2.0])
        self.ce_adversarial = nn.CrossEntropyLoss()  # kept for API compat

    def __call__(self, model_output, id_labels, gender_labels):
        """
        Args:
            model_output: dict with keys:
                id_logits, embedding, gender_logits, Fm, Fk
            id_labels:     [B] integer identity labels
            gender_labels: [B] integer gender labels

        Returns:
            dict with loss terms
        """
        # ── Identity CE (Fk-based embedding) ──────────────────────────────
        l_identity = self.ce_identity(model_output['id_logits'], id_labels)

        # ── Triplet (Fk-based embedding) ───────────────────────────────────
        l_triplet, triplet_stats = self.triplet(
            model_output['embedding'], id_labels
        )

        # ── Gender CE on Fm' (positive, class-weighted) ───────────────────
        gender_w = self.gender_weights.to(model_output['gender_logits'].device)
        l_gender = F.cross_entropy(
            model_output['gender_logits'], gender_labels, weight=gender_w
        )

        # ── Orthogonality loss (Fm ⊥ Fk) ──────────────────────────────────
        # Use pre-graph features for cleaner gradient signal
        l_orth = orthogonality_loss(
            model_output['Fm'], model_output['Fk']
        )

        # ── Weighted total ─────────────────────────────────────────────────
        total = (self.w_identity      * l_identity
               + self.w_triplet       * l_triplet
               + self.w_gender        * l_gender
               + self.w_orthogonality * l_orth)

        return {
            'total':         total,
            'identity':      l_identity,
            'triplet':       l_triplet,
            'gender':        l_gender,
            'adversarial':   l_orth,    # reuse key for logging continuity
            'mean_pos_dist': triplet_stats['mean_pos_dist'],
            'mean_neg_dist': triplet_stats['mean_neg_dist'],
        }
