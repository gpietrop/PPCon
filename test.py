import torch
from architecture import *


x = torch.rand([1, 7, 200])  # shape of the signal
print(x.shape)

model = Conv1dMed()
output = model(x.float())

print(output.shape)
