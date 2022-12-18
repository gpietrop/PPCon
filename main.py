import os
import argparse

import torch
from torch.utils.data import DataLoader

from train import train_model
from dataset import FloatDataset


# Setting the computation device
from utils import make_ds

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"We will use {device}")

# Create the parser
parser = argparse.ArgumentParser()
parser.add_argument('--training_folder', type=str, default="SUPERFLOAT", choices=["SUPERFLOAT", "CORIOLIS"])
parser.add_argument('--flag_toy', type=bool, default=False)
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--epochs', type=int, default=10**3)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--snaperiod', type=int, default=25)

# Parsing arguments
args = parser.parse_args()
training_folder = args.training_folder
flag_toy = 0
batch_size = args.batch_size
epochs = args.epochs
lr = args.lr
snaperiod = args.snaperiod

# Creating the correct dataframe according to the training folder
make_ds(training_folder, flag_complete=0, flag_toy=1)

if training_folder == "SUPERFLOAT":
    path_ds = os.getcwd() + "/ds/toy_ds_sf.csv" if flag_toy else os.getcwd() + "/ds/float_ds_sf.csv"
dataset = FloatDataset(path_ds)

train_frac = 0.8
train_size = int(train_frac * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

train_model(train_loader, val_loader, epoch=epochs, lr=lr, snaperiod=snaperiod, device=device)
