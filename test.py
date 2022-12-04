import torch
from conv1med import *
from mlp import *


x = torch.rand([1])  # shape of the signal
#print(x.shape)

model = MLPDay()

print(model.network[0].weight[0].item())

output = model(x.float())

output = output.unsqueeze(0).unsqueeze(0)
#print(output.shape)

y = torch.rand([1, 3, 200])

third_tensor = torch.cat((output, y), 1)
#print(third_tensor.shape)


print(output.shape)
