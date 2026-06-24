import torch.nn as nn

class MotionEncoder(nn.Module):
    def __init__(self, in_channels=1, channels=None):
        super().__init__()
        
        if channels is None:
            channels = [32, 64, 128, 256, 512]

        strides = [
            (1, 1, 1),
            (2, 2, 2),
            (1, 1, 1),
            (2, 2, 2),
            (1, 1, 1),
        ]

        in_dims  = [in_channels] + channels[:-1]
        out_dims = channels

        blocks = []
        for in_c, out_c, stride in zip(in_dims, out_dims, strides):
            blocks.append(
                nn.Sequential(
                    nn.Conv3d(in_c, out_c,
                              kernel_size=(3, 3, 3),
                              stride=stride,
                              padding=(1, 1, 1),
                              bias=False),
                    nn.BatchNorm3d(out_c),
                    nn.ReLU(inplace=True),
                )
            )

        self.encoder = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, motion):
        x = self.encoder(motion)
        x = self.pool(x)
        Fk = x.flatten(start_dim=1)
        return Fk
