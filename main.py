import os
from datetime import date
import argparse

import torch
from torch.utils.data import DataLoader

from train import train_model
from dataset import FloatDataset
from plot_profile import plot_profiles


# Setting the computation device
from utils import make_ds, save_ds_info

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"We will use {device}")

# ===== Create the parser
parser = argparse.ArgumentParser()
parser.add_argument('--training_folder', type=str, default="SUPERFLOAT", choices=["SUPERFLOAT", "CORIOLIS"])
parser.add_argument('--flag_toy', type=bool, default=False)
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--epochs', type=int, default=10**2)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--snaperiod', type=int, default=25)
parser.add_argument('--dropout_rate', type=float, default=0.1)

# ===== Parsing arguments
args = parser.parse_args()
training_folder = args.training_folder
flag_toy = 0
batch_size = args.batch_size
epochs = args.epochs  # args.epochs
lr = args.lr
snaperiod = args.snaperiod
dp_rate = args.dropout_rate

# ===== Printing information about the run
print(f"The dataset used is {training_folder}\nWe used a reduced version of the ds? {bool(flag_toy)}\n"
      f"The total number of epochs that will be performed is {epochs}")

# ===== Creating the correct dataframe according to the training folder
make_ds(training_folder, flag_complete=1, flag_toy=1)

if training_folder == "SUPERFLOAT":
    path_ds = os.getcwd() + "/ds/toy_ds_sf.csv" if flag_toy else os.getcwd() + "/ds/float_ds_sf.csv"
dataset = FloatDataset(path_ds)

train_frac = 0.8
train_size = int(train_frac * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

save_dir = os.getcwd() + "/results/"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
save_dir = save_dir + str(date.today())
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# ===== Saving models hyperparameters
save_ds_info(training_folder=training_folder, flag_toy=flag_toy, batch_size=batch_size, epochs=epochs, lr=lr,
             dp_rate=dp_rate, save_dir=save_dir)

# ===== train the model
train_model(train_loader, val_loader, epoch=epochs, lr=lr, dp_rate=dp_rate, snaperiod=snaperiod, save_dir=save_dir,
            device=device)

# ===== plot the results obtained on the validation set
plot_profiles(DataLoader(val_dataset, batch_size=1, shuffle=True), dir=save_dir, ep=epochs)
