import torch
import torch.nn as nn

cspdn53_config = [
    (32, 3, 1),
    ["D", 16, 512, 32, 32], #  Out Channel, Final Out Channel, Kernel Size, Strides
    (32, 3, 2),
    ["B", 1],
    (68, 3, 2),
    ["B", 2],
    (128, 3, 2),
    ["B", 8],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 4],
] 


dn53_config = [
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


# Downsample
class Downsample(nn.Module):
    def __init__(self,
        channels,
        final_channels,
        **kwargs):
        super().__init__()
        self.conv = CNNBlock(channels, final_channels,**kwargs)

    def forward(self, x):
        x_store, x = torch.tensor_split(x, 2, dim=1)
        x_store = self.conv(x_store)
        return x_store, x


class Concat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_store, x):
        x = torch.cat([x_store, x], dim=1)
        return x


class Darknet53(nn.Module):
    def __init__(self,
        in_channels=3,
        num_classes=20,
        csp=False):
        super().__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.csp = csp
        
        self.concat = Concat()
        self.final = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=2),
            nn.Softmax(dim=1),
            )


        self.layers = self._create_conv_layers()
        self.initialize_weights()


    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        if self.csp:
            print("Loading CSPDarknet53!")
            for module in cspdn53_config:
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
                    if module[0] == "B":
                        num_repeats = module[1]
                        layers.append(
                            ResidualBlock(
                                in_channels,
                                num_repeats=num_repeats
                            )
                        )
                    elif module[0] == "D":
                        out_channels, final_channels, kernel_size, stride = module[1:]
                        layers.append(
                            Downsample(
                                out_channels,
                                final_channels,
                                kernel_size=kernel_size,
                                stride=stride
                            )
                        )
                        in_channels = out_channels

        else:
            print('Loading Darknet53')
            for module in dn53_config:
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
                    if module[0] == "B":
                        num_repeats = module[1]
                        layers.append(
                            ResidualBlock(
                                in_channels,
                                num_repeats=num_repeats
                            )
                        )

        return layers


    def forward(self, x):
        if self.csp:
            for layer in self.layers:
                if isinstance(layer, Downsample):
                    x_store, x = layer(x)
                else:
                    x = layer(x)

            x = self.final(self.concat(x_store, x))
        else:
            for layer in self.layers:
                x = layer(x)

            x = self.final( x)         
        
        return x


    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    x = torch.randn(4,3,256,256)
    model = Darknet53(3, 2, csp=True)
    x = model(x)
    print(x.shape)