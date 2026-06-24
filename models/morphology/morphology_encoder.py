import torch.nn as nn

class MorphologyEncoder(nn.Module):
    def __init__(self, in_channels=1, channels=None):
        super().__init__()
        
        if channels is None:
            channels = [32, 64, 128, 256, 512]

        strides = [1, 2, 1, 2, 2]

        in_dims  = [in_channels] + channels[:-1] 
        out_dims = channels 

        blocks = []
        for in_c, out_c, stride in zip(in_dims, out_dims, strides):
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_c, out_c,
                              kernel_size=3, stride=stride,
                              padding=1, bias=False),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True),
                )
            )

        self.encoder = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, gei):
        x = self.encoder(gei)
        x = self.pool(x)
        Fm = x.flatten(start_dim=1)
        return Fm
