import torch.nn as nn
import numpy as np

def init_weights(layer, last_layer=False):
    if last_layer:
        f = 0.003
    else:
        if isinstance(layer, nn.Conv2d):
            f = 1 / np.sqrt(layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1])
        elif isinstance(layer, nn.Linear):
            f = 1 / np.sqrt(layer.in_features)

    nn.init.uniform_(layer.weight.data, -f, f)
    nn.init.uniform_(layer.bias.data, -f, f)