import torch
import torch.nn as nn
import torch.nn.functional as F
from losses.triplet import TripletLoss


class CombinedLoss:

    def __init__(self, w_identity=0.5, w_triplet=1.0, w_gender=0.3,
                 w_adversarial=0.3, triplet_margin=0.5, num_classes=None,
                 **kwargs):  # absorb unused arcface kwargs gracefully
        self.w_identity    = w_identity
        self.w_triplet     = w_triplet
        self.w_gender      = w_gender
        self.w_adversarial = w_adversarial

        # Plain CrossEntropy for identity — stable from scratch
        self.ce_identity = nn.CrossEntropyLoss()

        # Triplet with increased margin
        self.triplet = TripletLoss(margin=triplet_margin)

        # Weighted gender CE on Fm — Female weight doubled to prevent collapse
        # Previous [0.801, 1.330] was insufficient; stronger imbalance correction needed
        self.gender_weights = torch.tensor([1.0, 2.0])

        # Standard CE for adversarial gender on Fk
        self.ce_adversarial = nn.CrossEntropyLoss()

    def __call__(self, model_output, id_labels, gender_labels):
        """
        Args:
            model_output:  dict with keys: id_logits, embedding,
                           gender_logits, gender_logits_adv
            id_labels:     [B] integer identity labels (0-indexed)
            gender_labels: [B] integer gender labels

        Returns:
            dict with loss terms
        """
        # ── Identity CE ────────────────────────────────────────────────────
        l_identity = self.ce_identity(model_output['id_logits'], id_labels)

        # ── Triplet ────────────────────────────────────────────────────────
        l_triplet, triplet_stats = self.triplet(
            model_output['embedding'], id_labels
        )

        # ── Gender CE on Fm' (positive, class-weighted) ───────────────────
        gender_w = self.gender_weights.to(model_output['gender_logits'].device)
        l_gender = F.cross_entropy(
            model_output['gender_logits'], gender_labels, weight=gender_w
        )

        # ── Adversarial gender CE on Fk (via GRL) ─────────────────────────
        l_adversarial = self.ce_adversarial(
            model_output['gender_logits_adv'], gender_labels
        )

        # ── Weighted total ─────────────────────────────────────────────────
        total = (self.w_identity    * l_identity
               + self.w_triplet     * l_triplet
               + self.w_gender      * l_gender
               + self.w_adversarial * l_adversarial)

        return {
            'total':         total,
            'identity':      l_identity,
            'triplet':       l_triplet,
            'gender':        l_gender,
            'adversarial':   l_adversarial,
            'mean_pos_dist': triplet_stats['mean_pos_dist'],
            'mean_neg_dist': triplet_stats['mean_neg_dist'],
        }
