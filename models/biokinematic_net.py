import torch.nn as nn

from models.morphology.gei import generate_gei
from models.morphology.morphology_encoder import MorphologyEncoder
from models.motion.motion_generator import generate_motion
from models.motion.motion_encoder import MotionEncoder
from models.graph.bio_kinematic_graph import BioKinematicGraph
from models.heads.gender_head import GenderHead
from models.heads.identity_head import IdentityHead
from models.grl import GenderAdversary


class BioKinematicNet(nn.Module):
    """
    Full BioKinematicNet model.

    Disentangles morphology (static body shape) from motion (dynamic gait)
    and predicts both identity and gender from a silhouette sequence.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg: model config dict (from configs/model.yaml).
                 Expected structure:

                 cfg['morphology']['in_channels']       # int, default 1
                 cfg['morphology']['channels']          # list e.g. [32,64,128,256,512]

                 cfg['motion']['in_channels']           # int, default 1
                 cfg['motion']['channels']              # list e.g. [32,64,128,256,512]

                 cfg['graph']['node_dim']               # int, default 512
                 cfg['graph']['alpha_init']             # float, default 0.1

                 cfg['gender']['in_dim']                # int, default 512
                 cfg['gender']['hidden_dim']            # int, default 128
                 cfg['gender']['num_classes']           # int, default 2

                 cfg['identity']['num_classes']         # int, REQUIRED — training identities
                 cfg['identity']['hidden_dim']          # int, default 512
                 cfg['projection']['in_dim']            # int, default 512
                 cfg['projection']['out_dim']           # int, default 256
        """
        super().__init__()

        # ── Morphology branch ──────────────────────────────────────────────
        self.morph_encoder = MorphologyEncoder(
            in_channels=cfg['morphology']['in_channels'],
            channels=cfg['morphology']['channels'],
        )

        # ── Motion branch ──────────────────────────────────────────────────
        self.motion_encoder = MotionEncoder(
            in_channels=cfg['motion']['in_channels'],
            channels=cfg['motion']['channels'],
        )

        # ── Bio-kinematic graph ────────────────────────────────────────────
        self.graph = BioKinematicGraph(
            node_dim=cfg['graph']['node_dim'],
            alpha_init=cfg['graph']['alpha_init'],
        )

        # ── Gender head (morphology only) ──────────────────────────────────
        self.gender_head = GenderHead(
            in_dim=cfg['gender']['in_dim'],
            hidden_dim=cfg['gender']['hidden_dim'],
            num_classes=cfg['gender']['num_classes'],
        )

        # ── Gender adversary on Fk (via GRL) ──────────────────────────────
        # Fk → GRL → gender_adversary → adversarial gender loss
        # The GRL makes Fk uninformative about gender.
        self.gender_adversary = GenderAdversary(
            in_dim=cfg['gender']['in_dim'],
            hidden_dim=cfg['gender']['hidden_dim'],
            num_classes=cfg['gender']['num_classes'],
            lambda_=cfg['gender'].get('grl_lambda', 0.1),
        )

        # ── Identity head (both branches, fused) ───────────────────────────
        self.identity_head = IdentityHead(
            node_dim=cfg['projection']['in_dim'],
            proj_dim=cfg['projection']['out_dim'],
            hidden_dim=cfg['identity']['hidden_dim'],
            num_classes=cfg['identity']['num_classes'],
        )

    def forward(self, x, mode='train'):
        """
        Args:
            x:    [B, T, 1, 224, 224] — raw silhouette sequence
            mode: one of 'train' | 'inference' | 'eval'

                  'train'     — full output dict including intermediates.
                                Used by the trainer every batch.

                  'inference' — embedding only [B, 512].
                                Used for gallery matching at test time.
                                Fastest path — no unnecessary computation
                                in the output stage.

                  'eval'      — embedding + gender_logits.
                                Used when evaluating gender accuracy on
                                the test set, or running analysis scripts
                                that need both outputs without the full
                                training intermediates (Fm, Fk, etc.).

        Returns:
            mode='train':
                dict with keys: gender_logits, id_logits, embedding,
                                Fm, Fk, Fm_prime, Fk_prime
            mode='inference':
                embedding [B, 512]
            mode='eval':
                dict with keys: embedding, gender_logits
        """
        assert mode in ('train', 'inference', 'eval'), \
            f"mode must be 'train', 'inference', or 'eval', got '{mode}'"

        # ── Step 1: Generate GEI and motion volume in parallel ─────────────
        # Both operate on the raw input sequence x.
        # They share no weights — the split happens here.

        # GEI: average over T frames → static body shape
        # [B, T, 1, 224, 224] → [B, 1, 224, 224]
        gei = generate_gei(x)

        # Motion: absolute frame differences → dynamic movement
        # [B, T, 1, 224, 224] → [B, 1, T-1, 224, 224]
        motion = generate_motion(x)

        # ── Step 2: Encode each branch independently ───────────────────────

        # Morphology CNN: GEI → compact body shape feature
        # [B, 1, 224, 224] → [B, 512]
        Fm = self.morph_encoder(gei)

        # Motion 3D CNN: motion volume → compact gait dynamics feature
        # [B, 1, T-1, 224, 224] → [B, 512]
        Fk = self.motion_encoder(motion)

        # ── Step 3: Bio-kinematic graph interaction ────────────────────────
        # Single round of residual message passing.
        # Fm' = Fm + α·Wm(Fk)  — morphology receives motion context
        # Fk' = Fk + α·Wk(Fm)  — motion receives morphology context
        # [B, 512], [B, 512] → [B, 512], [B, 512]
        Fm_prime, Fk_prime = self.graph(Fm, Fk)

        # ── Step 4: Gender head (morphology only) ─────────────────────────
        # Routes ONLY Fm' — enforces the hypothesis that gender is
        # determined by body shape, not by how you walk.
        # [B, 512] → [B, 2]
        gender_logits = self.gender_head(Fm_prime)

        # ── Step 4b: Gender adversary on Fk ───────────────────────────────
        # Fk → GRL → adversary — makes Fk uninformative about gender.
        # GRL reverses gradients: adversary tries to predict gender,
        # Fk tries to fool it → gender removed from motion features.
        gender_logits_adv = self.gender_adversary(Fk_prime)

        # ── Step 5: Identity head (both branches) ─────────────────────────
        # Routes both Fm' and Fk' — identity depends on shape AND motion.
        # embedding [B, 512] — pre-BNNeck, for triplet loss + retrieval
        # id_logits [B, num_classes] — post-BNNeck, for CE loss
        embedding, id_logits = self.identity_head(Fm_prime, Fk_prime)

        # ── Output ────────────────────────────────────────────────────────
        if mode == 'inference':
            # Gallery matching: nearest-neighbour search in embedding space.
            # id_logits and gender_logits are not needed here.
            return embedding

        if mode == 'eval':
            # Test-set evaluation: need embedding for retrieval metrics
            # (Rank-1, mAP) AND gender_logits for gender accuracy.
            # No training intermediates needed.
            return {
                'embedding':      embedding,      # [B, 512]
                'gender_logits':  gender_logits,  # [B, 2]
            }

        # mode == 'train'
        # Trainer uses gender_logits, id_logits, embedding for losses.
        # Analysis scripts use Fm, Fk, Fm_prime, Fk_prime for
        # disentanglement verification.
        return {
            'gender_logits':     gender_logits,      # [B, 2]
            'gender_logits_adv': gender_logits_adv,  # [B, 2] adversary on Fk
            'id_logits':         id_logits,           # [B, num_classes] (bn_feat)
            'embedding':         embedding,           # [B, 512]
            'Fm':                Fm,                  # [B, 512] pre-graph
            'Fk':                Fk,                  # [B, 512] pre-graph
            'Fm_prime':          Fm_prime,            # [B, 512] post-graph
            'Fk_prime':          Fk_prime,            # [B, 512] post-graph
        }

    def get_graph_stats(self, x):
        """
        Convenience method for analysis scripts.
        Runs a forward pass and returns the graph interaction statistics.

        Shows how strongly morphology and motion are influencing each other
        at the current training state.

        Args:
            x: [B, T, 1, 224, 224]

        Returns:
            dict with keys: alpha, motion_to_morph, morph_to_motion
        """
        gei    = generate_gei(x)
        motion = generate_motion(x)
        Fm     = self.morph_encoder(gei)
        Fk     = self.motion_encoder(motion)
        return self.graph.message_stats(Fm, Fk)

    def set_grl_lambda(self, lambda_):
        """
        Update GRL strength during training.
        Call this each epoch to gradually increase adversarial pressure.
        """
        self.gender_adversary.set_lambda(lambda_)

    def count_parameters(self):
        """
        Returns parameter count broken down by component.
        Useful for understanding where the model's capacity lives.
        """
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
