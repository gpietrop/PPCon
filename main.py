import os

import torch
from torch.utils.data import DataLoader, Dataset

from make_float_ds import make_pandas_df
from train import train_model
from dataset import FloatDataset


# make_pandas_df(os.getcwd() + '/FLOAT_BIO/data/Float_Index.txt')
path_float = os.getcwd() + "/float_ds.csv"
dataset = FloatDataset(path_float)

train_frac = 0.8
batch_size = 1

train_size = int(train_frac * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# for year, day_rad, lat, lon, temp, psal, doxy, label in train_loader:
#    print(label)
train_model(train_loader, val_loader, 100)

