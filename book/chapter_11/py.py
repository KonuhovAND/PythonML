import torch
import torch.nn as nn

layer = nn.Linear(40, 10)
nn.init.kaiming_uniform_(layer.weight)
nn.init.zeros_(layer.bias)
print(layer.bias.data)
