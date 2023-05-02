import os

import pandas as pd
import netCDF4 as nc
from torch.utils.data import DataLoader

from dataset_with_float_names import FloatDataset
from discretization import *
import numpy as np


pres = np.arange(0, dict_max_pressure["NITRATE"], dict_interval["NITRATE"])
print(pres)
print(len(pres))

variable, mode = "NITRATE", "test"
path_df = f"/home/gpietropolli/Desktop/canyon-float/ds/clustering/ds_sf_clustering.csv"
dataset = FloatDataset(path_df)
my_ds = DataLoader(dataset, shuffle=True)

for sample in my_ds:
    # print("hi")
    year, day_rad, lat, lon, temp, psal, doxy, nitrate, chla, BBP700, name_float = sample
    # print(year)
    sub_dir = name_float[0]
    main_dir = name_float[0][2:-4]
    path_superfloat = f"/home/gpietropolli/Desktop/canyon-float/ds/SUPERFLOAT/{main_dir}/{sub_dir}.nc"
    ds = nc.Dataset(path_superfloat)
    # print(ds)

    break

data = pd.read_csv(path_df)
# print(data)

data_top = list(data.columns)[1][-21:]

# print(data_top)

path_ds = "/home/gpietropolli/Desktop/canyon-float/ds/SUPERFLOAT_PPCon/6901467/MR6901467_179.nc"
ds = nc.Dataset(path_ds)
print(len(ds["PHOSPHATE_PPCon"][:]))
# print(ds)
