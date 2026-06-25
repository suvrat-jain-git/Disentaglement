import torch
import torch.nn as nn
from losses.triplet import TripletLoss


class CombinedLoss:
    """
    Weighted sum of identity CE, triplet, and gender CE losses.
    """

    def __init__(self, w_identity=1.0, w_triplet=1.0, w_gender=0.5,
                 triplet_margin=0.3, num_classes=None):
        """
        Args:
            w_identity:     weight for identity CrossEntropy loss
            w_triplet:      weight for triplet loss
            w_gender:       weight for gender CrossEntropy loss
            triplet_margin: margin for TripletLoss
            num_classes:    unused here, kept for API clarity
        """
        self.w_identity = w_identity
        self.w_triplet  = w_triplet
        self.w_gender   = w_gender

        # CrossEntropyLoss for identity classification.
        # Operates on id_logits [B, num_classes] and integer labels [B].
        # Internally applies log-softmax + NLL — expects raw logits, not probs.
        self.ce_identity = nn.CrossEntropyLoss()

        # CrossEntropyLoss for gender classification with class weights.
        # FVG-B training set: ~62% Male, ~38% Female.
        # Without weighting the gender head collapses to predicting
        # the majority class. Weights are inversely proportional to
        # class frequency: w_c = total / (num_classes * count_c)
        # Male (0):   117 / (2 * 73) = 0.801
        # Female (1): 117 / (2 * 44) = 1.330
        # These are computed from the actual training split.
        # If your split changes, update these weights accordingly.
        # Gender class weights — stored and moved to device at call time.
        # Male (0): 0.801, Female (1): 1.330
        # Computed from FVG-B training split: 62.4% Male, 37.6% Female.
        # w_c = total / (num_classes * count_c)
        self.gender_weights = torch.tensor([0.801, 1.330])
        self.ce_gender = nn.CrossEntropyLoss()  # weights applied in __call__

        # Batch hard triplet loss.
        # Operates on the pre-BNNeck embedding [B, D] — the metric space
        # shaped during training. Never use the post-BNNeck features here.
        self.triplet = TripletLoss(margin=triplet_margin)

    def __call__(self, model_output, id_labels, gender_labels):
        """
        Args:
            model_output:   dict from BioKinematicNet.forward(mode='train')
                            Required keys: 'id_logits', 'embedding', 'gender_logits'
            id_labels:      [B] — integer identity labels (0-indexed)
            gender_labels:  [B] — integer gender labels (0=Male, 1=Female)

        Returns:
            losses: dict with keys:
                'total'    — weighted sum, backpropagate this
                'identity' — unweighted CE loss, for logging
                'triplet'  — unweighted triplet loss, for logging
                'gender'   — unweighted CE loss, for logging
        """
        # ── Identity CE loss ───────────────────────────────────────────────
        # id_logits: [B, num_classes] — post-BNNeck classifier output
        # Trains the classifier head to discriminate training identities.
        l_identity = self.ce_identity(model_output['id_logits'], id_labels)

        # ── Triplet loss ───────────────────────────────────────────────────
        # embedding: [B, 512] — pre-BNNeck metric space embedding
        # Shapes the embedding space so same-identity embeddings cluster
        # and different-identity embeddings push apart.
        l_triplet, triplet_stats = self.triplet(
            model_output['embedding'], id_labels
        )

        # ── Gender CE loss ─────────────────────────────────────────────────
        # gender_logits: [B, 2] — morphology branch gender prediction
        # Regularises the morphology branch to encode gender-relevant
        # body structure. This is the disentanglement supervision signal.
        # Move gender weights to same device as logits at call time
        gender_w = self.gender_weights.to(model_output['gender_logits'].device)
        l_gender = nn.functional.cross_entropy(
            model_output['gender_logits'], gender_labels, weight=gender_w
        )

        # ── Weighted total ─────────────────────────────────────────────────
        total = (self.w_identity * l_identity
               + self.w_triplet  * l_triplet
               + self.w_gender   * l_gender)

        return {
            'total':    total,        # backpropagate this
            'identity': l_identity,   # log individually
            'triplet':  l_triplet,    # log individually
            'gender':   l_gender,     # log individually
            # Triplet stats for logging — tells you about embedding geometry
            'mean_pos_dist': triplet_stats['mean_pos_dist'],
            'mean_neg_dist': triplet_stats['mean_neg_dist'],
        }
