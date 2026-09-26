import torch
import torch.nn as nn

layer = nn.Linear(40, 10)
layer.weight.data *= 6**0.5
torch.zero_(layer.bias.data)
