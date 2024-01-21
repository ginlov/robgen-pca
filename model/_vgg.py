import torch
from torch import nn
from typing import Any, Dict, List, Union, cast
from model.modified_layer import ModifiedConv2d, ModifiedMaxPool2d, ModifiedLinear,ModifiedAdaptiveAvgPool2d

cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}

class VGG(nn.Module):
    def __init__(
        self, features: nn.Module, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.features = features
        self.avgpool = ModifiedAdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            ModifiedLinear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            ModifiedLinear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            ModifiedLinear(4096, num_classes)
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def make_layers(cfg: List[Union[str, int]], norm_layer=None) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = ModifiedConv2d(in_channels, v, kernel_size=3, padding=1)
            if norm_layer is not None:
                if norm_layer == nn.BatchNorm2d:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                elif norm_layer == nn.GroupNorm:
                    layers += [conv2d, nn.GroupNorm(int(v/2), v), nn.ReLU(inplace=True)]
                elif norm_layer == nn.LayerNorm:
                    layers += [conv2d, nn.GroupNorm(1, v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def _vgg(cfg: str, norm_layer, **kwargs: Any) -> VGG:
    model = VGG(make_layers(cfgs[cfg], norm_layer=norm_layer), **kwargs)
    return model
