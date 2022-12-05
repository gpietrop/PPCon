import torch
import os
import numpy as np
import pandas as pd
from conv1med import *
from mlp import *

x = torch.rand([1])  # shape of the signal
# print(x.shape)

model = MLPDay()

# print(model.network[0].weight[0].item())

output = model(x.float())

output = output.unsqueeze(0).unsqueeze(0)
# print(output.shape)

y = torch.rand([1, 3, 200])

third_tensor = torch.cat((output, y), 1)
# print(third_tensor.shape)
# print(output.shape)


def from_string_to_tensor(string):
    # string = df.iloc[6, 1][8:-2]
    string = string[8:-2]
    string = string.split(",")
    out = torch.zeros(200)
    for ind in range(len(string)):
        out[ind] = torch.tensor(float(string[ind]))
    return out

path = os.getcwd() + "/float_ds.csv"
df = pd.read_csv(os.getcwd() + "/float_ds.csv")
# a = df.iloc[:, 1].tolist()
a = df.iloc[:, :]


from dataset import FloatDataset

x = FloatDataset(path)
print(x.df)

# temp = a[4]
# temp = float(temp)
temp = df.iloc[6, 1][8:-2]
temp = temp.split(",")
# temp = torch.tensor(temp)
tens = torch.zeros(200)
for index in range(len(temp)):
    tens[index] = torch.tensor(float(temp[index]))




