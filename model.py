import torch
import torch.nn as nn

config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],
]

# Basic CNN Block
class CNNBlock(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        bn_act=True,
        **kwaargs):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwaargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.bn_act = bn_act

    def forward(self, x):
        if self.bn_act:
            x = self.conv(x)
            x = self.bn(x)
            x = self.leaky(x)
        else:
            x = self.conv(x)
        
        return x
            

# Residual Block (+)
class ResidualBlock(nn.Module):
    def __init__(self,
        channels,
        use_residual=True,
        num_repeats=1):
        super().__init__()

        self.use_residual = use_residual
        self.num_repeats = num_repeats

        self.layers = nn.ModuleList()
        for _ in range(num_repeats):
            self.layers += [nn.Sequential(
                    CNNBlock(channels, channels//2, kernel_size=1),
                    CNNBlock(channels//2, channels, kernel_size=3, padding=1)
                )
            ]
        
        
    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = layer(x) + x
            else:
                x = layer(x)
        
        return x


class Darknet53(nn.Module):
    def __init__(self,
        in_channels=3,
        num_classes=20):
        super().__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.final = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(), 
            nn.Linear(in_features=1024, out_features=2),
            nn.Softmax(dim=1),
            )
        self.layers = self._create_conv_layers()


    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding = 1 if kernel_size == 3 else 0
                    )
                )
                in_channels = out_channels

            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(
                    ResidualBlock(
                        in_channels,
                        num_repeats=num_repeats
                    )
                )
            

        return layers


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        x = self.final(x)
        return x

if __name__ == '__main__':
    x = torch.randn(4,3,256,256) 
    model = Darknet53()
    x = model(x)
    print(x.shape)