import os
import random

import numpy as np
import pandas as pd
import torch

from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import FloatDataset
from utils import upload_and_evaluate_model, from_day_rad_to_day

dict_max_pressure = {"NITRATE": 1000,
                     "CHLA": 200,
                     "BBP700": 200}
dict_var_name = {"NITRATE": "Nitrate",
                 "CHLA": "Chlorophyll",
                 "BBP700": "BBP700"}
dict_unit_measure = {"NITRATE": r"$mmol \ m^{-3}$",
                     "CHLA": r"$mg \ m^{-3}$",
                     "BBP700": r"$m^{-1}$"}


def moving_average(data, window_size):
    if window_size % 2 == 0:
        window_size += 1  # Ensure window size is odd for symmetry
    pad_size = window_size // 2
    padded_data = np.pad(data, pad_size, mode='edge')
    cumsum_vec = np.cumsum(np.insert(padded_data, 0, 0))
    return (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size


def get_reconstruction_user(variable, date_model, epoch_model, mode, data=None):
    # Determine the dataset source
    if data is not None:
        # If data is provided as tensors, create a TensorDataset directly
        dataset = TensorDataset(*data)
    else:
        path_float = os.getcwd() + f"/../ds/{variable}/float_ds_sf_{mode}.csv"
        if mode == "all":
            path_float = os.getcwd() + f"/../ds/{variable}/float_ds_sf.csv"
        dataset = FloatDataset(path_df=path_float)  # Assuming FloatDataset handles CSV loading and tensor conversion

    ds = DataLoader(dataset, shuffle=True)

    dir_model = os.getcwd() + f"/../results/{variable}/{date_model}/model"
    info = pd.read_csv(os.getcwd() + f"/../results/{variable}/{date_model}/info.csv")

    # Upload and evaluate the model
    model_day, model_year, model_lat, model_lon, model = upload_and_evaluate_model(
        dir_model=dir_model, info_model=info, ep=epoch_model
    )

    lat_list = []
    lon_list = []
    day_rad_list = []
    measured_var_list = []
    generated_var_list = []

    for sample in ds:
        year, day_rad, lat, lon, temp, psal, doxy, measured_var = sample

        output_day = model_day(day_rad.unsqueeze(1))
        output_year = model_year(year.unsqueeze(1))
        output_lat = model_lat(lat.unsqueeze(1))
        output_lon = model_lon(lon.unsqueeze(1))

        output_day = torch.transpose(output_day.unsqueeze(0), 0, 1)
        output_year = torch.transpose(output_year.unsqueeze(0), 0, 1)
        output_lat = torch.transpose(output_lat.unsqueeze(0), 0, 1)
        output_lon = torch.transpose(output_lon.unsqueeze(0), 0, 1)
        temp = torch.transpose(temp.unsqueeze(0), 0, 1)
        psal = torch.transpose(psal.unsqueeze(0), 0, 1)
        doxy = torch.transpose(doxy.unsqueeze(0), 0, 1)

        x = torch.cat((output_day, output_year, output_lat, output_lon, temp, psal, doxy), 1)
        generated_var = model(x.float())
        generated_var = generated_var.detach()

        lat_list.append(lat.item())
        lon_list.append(lon.item())
        day_rad_list.append(day_rad.item())

        if variable == "NITRATE":
            generated_var = torch.squeeze(generated_var)[:-10]
            measured_var = torch.squeeze(measured_var)[:-10]
        if variable == "BBP700":
            generated_var = torch.squeeze(generated_var) / 1000
            measured_var = torch.squeeze(measured_var) / 1000
        else:
            generated_var = torch.squeeze(generated_var)
            measured_var = torch.squeeze(measured_var)

        generated_var_list.append(generated_var)
        measured_var_list.append(measured_var)

    return lat_list, lon_list, day_rad_list, generated_var_list, measured_var_list
