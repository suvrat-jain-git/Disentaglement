import torch.nn as nn

class GenderHead(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=128, num_classes=2):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, Fm_prime):
        return self.head(Fm_prime)
