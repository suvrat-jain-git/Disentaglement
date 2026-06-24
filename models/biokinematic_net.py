import torch.nn as nn

from models.morphology.gei import generate_gei
from models.morphology.morphology_encoder import MorphologyEncoder
from models.motion.motion_generator import generate_motion
from models.motion.motion_encoder import MotionEncoder
from models.graph.bio_kinematic_graph import BioKinematicGraph
from models.heads.gender_head import GenderHead
from models.heads.identity_head import IdentityHead

class BioKinematicNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.morph_encoder = MorphologyEncoder(
            in_channels=cfg['morphology']['in_channels'],
            channels=cfg['morphology']['channels'],
        )

        self.motion_encoder = MotionEncoder(
            in_channels=cfg['motion']['in_channels'],
            channels=cfg['motion']['channels'],
        )

        self.graph = BioKinematicGraph(
            node_dim=cfg['graph']['node_dim'],
            alpha_init=cfg['graph']['alpha_init'],
        )

        self.gender_head = GenderHead(
            in_dim=cfg['gender']['in_dim'],
            hidden_dim=cfg['gender']['hidden_dim'],
            num_classes=cfg['gender']['num_classes'],
        )

        self.identity_head = IdentityHead(
            node_dim=cfg['projection']['in_dim'],
            proj_dim=cfg['projection']['out_dim'],
            hidden_dim=cfg['identity']['hidden_dim'],
            num_classes=cfg['identity']['num_classes'],
        )

    def forward(self, x, mode='train'):
        assert mode in ('train', 'inference', 'eval'), \
            f"mode must be 'train', 'inference', or 'eval', got '{mode}'"

        gei = generate_gei(x)
        motion = generate_motion(x)

        Fm = self.morph_encoder(gei)

        Fk = self.motion_encoder(motion)

        Fm_prime, Fk_prime = self.graph(Fm, Fk)

        gender_logits = self.gender_head(Fm_prime)

        embedding, id_logits = self.identity_head(Fm_prime, Fk_prime)

        if mode == 'inference':
            return embedding

        if mode == 'eval':
            return {
                'embedding':      embedding,      
                'gender_logits':  gender_logits,
            }

        return {
            'gender_logits': gender_logits,   
            'id_logits':     id_logits,        
            'embedding':     embedding,        
            'Fm':            Fm,               
            'Fk':            Fk,               
            'Fm_prime':      Fm_prime,         
            'Fk_prime':      Fk_prime,         
        }

    def get_graph_stats(self, x):
        gei    = generate_gei(x)
        motion = generate_motion(x)
        Fm     = self.morph_encoder(gei)
        Fk     = self.motion_encoder(motion)
        return self.graph.message_stats(Fm, Fk)

    def count_parameters(self):
        def count(module):
            return sum(p.numel() for p in module.parameters())

        breakdown = {
            'morph_encoder':  count(self.morph_encoder),
            'motion_encoder': count(self.motion_encoder),
            'graph':          count(self.graph),
            'gender_head':    count(self.gender_head),
            'identity_head':  count(self.identity_head),
        }
        breakdown['total'] = sum(breakdown.values())
        return breakdown
