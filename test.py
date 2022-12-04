import torch
from conv1med import *
from mlp import *


x = torch.rand([1])  # shape of the signal
print(x.shape)

model = MLPDay()
output = model(x.float())

print(output.shape)
