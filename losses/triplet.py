import torch
import torch.nn as nn

class TripletLoss:
    def __init__(self, margin=0.3):
        self.margin = margin

    def _pairwise_distances(self, embeddings):
        sq_norm = (embeddings ** 2).sum(dim=1)
        sq_norm_row = sq_norm.unsqueeze(1) 
        sq_norm_col = sq_norm.unsqueeze(0)  
        dot = torch.mm(embeddings, embeddings.t())
        dist_sq = (sq_norm_row + sq_norm_col - 2.0 * dot).clamp(min=0.0)
        dist = (dist_sq + 1e-12).sqrt()
        return dist

    def __call__(self, embeddings, labels):
        B = embeddings.size(0)

        dist = self._pairwise_distances(embeddings)

        labels_col = labels.unsqueeze(0)
        labels_row = labels.unsqueeze(1)  
        labels_equal = labels_row == labels_col

        eye = torch.eye(B, dtype=torch.bool, device=embeddings.device)
        positive_mask = labels_equal & ~eye

        negative_mask = ~labels_equal

        pos_dist = dist * positive_mask.float()
        hardest_pos_dist = pos_dist.max(dim=1).values  

        neg_dist = dist + (~negative_mask).float() * 1e9
        hardest_neg_dist = neg_dist.min(dim=1).values     

        triplet_loss = (hardest_pos_dist - hardest_neg_dist + self.margin)
        triplet_loss = triplet_loss.clamp(min=0.0).mean()

        with torch.no_grad():
            mean_pos = (dist * positive_mask.float()).sum() / positive_mask.float().sum().clamp(min=1)
            mean_neg = (dist * negative_mask.float()).sum() / negative_mask.float().sum().clamp(min=1)

        stats = {
            'mean_pos_dist': mean_pos.item(),
            'mean_neg_dist': mean_neg.item(),
        }

        return triplet_loss, stats
