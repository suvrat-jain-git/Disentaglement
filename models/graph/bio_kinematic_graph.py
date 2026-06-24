import torch
import torch.nn as nn

class BioKinematicGraph(nn.Module):
    def __init__(self, node_dim=512, alpha_init=0.1):
        super().__init__()

        self.Wm = nn.Linear(node_dim, node_dim)

        self.Wk = nn.Linear(node_dim, node_dim)

        self.alpha = nn.Parameter(torch.tensor(alpha_init))

    def forward(self, Fm, Fk):
        Fm_prime = Fm + self.alpha * self.Wm(Fk)

        Fk_prime = Fk + self.alpha * self.Wk(Fm)

        return Fm_prime, Fk_prime

    def message_stats(self, Fm, Fk):
        with torch.no_grad():
            msg_to_morph = self.alpha * self.Wm(Fk)
            msg_to_motion = self.alpha * self.Wk(Fm)  

            ratio_m = (msg_to_morph.norm(dim=1) / Fm.norm(dim=1)).mean().item()
            ratio_k = (msg_to_motion.norm(dim=1) / Fk.norm(dim=1)).mean().item()

        return {
            'alpha':           self.alpha.item(),
            'motion_to_morph': ratio_m,
            'morph_to_motion': ratio_k,
        }
