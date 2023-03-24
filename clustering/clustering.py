import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from utils_clustering import make_ds
from dataset_clustering import FloatDataset

make_ds("SUPERFLOAT")
ds_enhanced_path = os.getcwd() + "/../ds/clustering/ds_sf_clustering_enhanced.csv"

dataset = FloatDataset(ds_enhanced_path)
ds = DataLoader(dataset, shuffle=True)

for year, day_rad, lat, lon, temp, psal, doxy, nitrate, chla, BBP700, name_float in ds:

    nitrate = nitrate[:, 5:150]
    BBP700 = BBP700 / 1000

    print(nitrate)

    plt.plot(BBP700[0, :].detach().numpy())
    # plt.plot(output[0, 0, :].detach().numpy(), label="measured")
    plt.legend()
    # plt.savefig(dir_profile + f"/profile_{year}_{day_rad}_{round(lat, 2)}_{round(lon, 2)}.png")
    plt.show()
    plt.close()
