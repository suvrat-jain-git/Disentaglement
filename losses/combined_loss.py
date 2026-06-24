import torch.nn as nn
from losses.triplet import TripletLoss

class CombinedLoss:
    def __init__(self, w_identity=1.0, w_triplet=1.0, w_gender=0.5,
                 triplet_margin=0.3, num_classes=None):
        
        self.w_identity = w_identity
        self.w_triplet  = w_triplet
        self.w_gender   = w_gender

        self.ce_identity = nn.CrossEntropyLoss()

        self.ce_gender = nn.CrossEntropyLoss()

        self.triplet = TripletLoss(margin=triplet_margin)

    def __call__(self, model_output, id_labels, gender_labels):

        l_identity = self.ce_identity(model_output['id_logits'], id_labels)

        l_triplet, triplet_stats = self.triplet(
            model_output['embedding'], id_labels
        )

        l_gender = self.ce_gender(model_output['gender_logits'], gender_labels)

        total = (self.w_identity * l_identity
               + self.w_triplet  * l_triplet
               + self.w_gender   * l_gender)

        return {
            'total':    total,        
            'identity': l_identity,   
            'triplet':  l_triplet,    
            'gender':   l_gender,     
            'mean_pos_dist': triplet_stats['mean_pos_dist'],
            'mean_neg_dist': triplet_stats['mean_neg_dist'],
        }
