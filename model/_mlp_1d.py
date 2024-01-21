import torch

from torch import nn
from typing import Union, List, Any, cast
from model.modified_layer import ModifiedLinear 

class MLP_1d(nn.Module):
    def __init__(self, 
                 in_features: int,
                 cfg: List[Union[str, int]], 
                 norm_layer = None,
                 num_classes: int = 1000) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        _in_features = in_features
        for v in cfg:
            v = cast(int, v)
            linear_layer = ModifiedLinear(_in_features, v)
            if norm_layer is not None:
                if norm_layer == nn.BatchNorm1d:
                    layers += [linear_layer, nn.BatchNorm1d(v), nn.ReLU(inplace=True)]
                elif norm_layer == nn.GroupNorm:
                    layers += [linear_layer, nn.GroupNorm(int(v/2), v), nn.ReLU(inplace=True)]
            else:
                layers += [linear_layer, nn.ReLU(inplace=True)]
            _in_features = v
        self.features_extractor = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            ModifiedLinear(_in_features, 32),
            nn.ReLU(inplace=True),
            ModifiedLinear(32, num_classes)
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        #Comment the above and Uncomment the following line for calculate Local_lipschitz
        #torch.view(-1, 3*32*32)
        x = self.features_extractor(x)
        x = self.classifier(x)
        return x

def _mlp_1d(in_features: int, cfg: List[Union[int, str]], norm_layer, num_classes: int, **kwargs: Any) -> MLP_1d:
    model = MLP_1d(in_features, cfg, norm_layer, num_classes)
    return model
