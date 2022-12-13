import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from make_ds import make_pandas_df, make_pandas_toy_df
from train import train_model
from dataset import FloatDataset


# make_pandas_df(os.getcwd() + '/FLOAT_BIO/data/Float_Index.txt')
# make_pandas_toy_df(os.getcwd() + '/FLOAT_BIO/data/Float_Index.txt')

path_float = os.getcwd() + "/toy_ds.csv"
dataset = FloatDataset(path_float)
# a = pd.read_csv(path_float)
# print(a)
# print(a.iloc[-1,  191])


train_frac = 0.8
batch_size = 12

train_size = int(train_frac * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# Setting the computation device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"We will use {device}")

train_model(train_loader, val_loader, 100, device)
