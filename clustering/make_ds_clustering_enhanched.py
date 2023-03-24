import os
import pandas as pd
import torch
from torch.utils.data import DataLoader

from architecture.conv1med_dp import Conv1dMed
from architecture.mlp import MLPDay, MLPYear, MLPLon, MLPLat
from dataset_clustering import FloatDataset


def upload_and_evaluate_model(dir_model, info_model, ep):
    dp_rate = info_model['dp_rate'].item()

    # Path of the saved models
    path_model_day = dir_model + "/model_day_" + str(ep) + ".pt"
    path_model_year = dir_model + "/model_year_" + str(ep) + ".pt"
    path_model_lon = dir_model + "/model_lon_" + str(ep) + ".pt"
    path_model_lat = dir_model + "/model_lat_" + str(ep) + ".pt"
    path_model_conv = dir_model + "/model_conv_" + str(ep) + ".pt"

    # Upload and evaluate all the models necessary
    model_day = MLPDay()
    model_day.load_state_dict(torch.load(path_model_day,
                                         map_location=torch.device('cpu')))
    model_day.eval()

    model_year = MLPYear()
    model_year.load_state_dict(torch.load(path_model_year,
                                          map_location=torch.device('cpu')))
    model_year.eval()

    model_lat = MLPLat()
    model_lat.load_state_dict(torch.load(path_model_lat,
                                         map_location=torch.device('cpu')))
    model_lat.eval()

    model_lon = MLPLon()
    model_lon.load_state_dict(torch.load(path_model_lon,
                                         map_location=torch.device('cpu')))
    model_lon.eval()

    model = Conv1dMed(dp_rate=dp_rate)
    model.load_state_dict(torch.load(path_model_conv,
                                     map_location=torch.device('cpu')))
    model.eval()

    return model_day, model_year, model_lat, model_lon, model


def get_output(sample, model_day, model_year, model_lat, model_lon, model):
    year, day_rad, lat, lon, temp, psal, doxy, _, _, _ = sample

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
    output_model = model(x.float())

    return output_model


# Where to search the model
dir = "results"

date_nitrate = "2023-03-06_"
ep_nitrate = 50

date_chla = "2023-03-15"
ep_chla = 1000

date_BBP700 = "2023-03-16"
ep_BBP700 = 100

# Upload the input ds
path_float = f"/home/gpietropolli/Desktop/canyon-float/ds/clustering/ds_sf_clustering.csv"
dataset = FloatDataset(path_float)
ds = DataLoader(dataset, shuffle=True)

dir_model_nitrate = os.getcwd() + f"/../{dir}/NITRATE/{date_nitrate}/model"
dir_model_chla = os.getcwd() + f"/../{dir}/CHLA/{date_chla}/model"
dir_model_BBP700 = os.getcwd() + f"/../{dir}/BBP700/{date_BBP700}/model"

# Upload the input information
info_nitrate = pd.read_csv(os.getcwd() + f"/../{dir}/NITRATE/{date_nitrate}/info.csv")
info_chla = pd.read_csv(os.getcwd() + f"/../{dir}/CHLA/{date_chla}/info.csv")
info_BBP700 = pd.read_csv(os.getcwd() + f"/../{dir}/BBP700/{date_BBP700}/info.csv")

# Upload and evaluate the model
model_day_nitrate, model_year_nitrate, model_lat_nitrate, model_lon_nitrate, model_nitrate = upload_and_evaluate_model(
    dir_model=dir_model_nitrate, info_model=info_nitrate, ep=ep_nitrate)
model_day_chla, model_year_chla, model_lat_chla, model_lon_chla, model_chla = upload_and_evaluate_model(
    dir_model=dir_model_chla, info_model=info_chla, ep=ep_chla)
model_day_BBP700, model_year_BBP700, model_lat_BBP700, model_lon_BBP700, model_BBP700 = upload_and_evaluate_model(
    dir_model=dir_model_BBP700, info_model=info_BBP700, ep=ep_BBP700)

dict_ds_enhanced = dict()

for sample in ds:
    year, day_rad, lat, lon, temp, psal, doxy, nitrate, chla, BBP700, name_float = sample
    sample_prediction = sample[:-1]

    # print(temp)
    # print(torch.squeeze(temp))
    # break
    if torch.count_nonzero(nitrate) == 0:
        # print(nitrate)
        nitrate = get_output(sample=sample_prediction, model_day=model_day_nitrate, model_year=model_year_nitrate,
                             model_lat=model_lat_nitrate, model_lon=model_lon_nitrate, model=model_nitrate)
        # nitrate = nitrate.detach().numpy()
        # print(nitrate.shape)
        nitrate = nitrate.detach()
        # print(nitrate.shape)

    if torch.count_nonzero(chla) == 0:
        chla = get_output(sample=sample_prediction, model_day=model_day_chla, model_year=model_year_chla,
                          model_lat=model_lat_chla, model_lon=model_lon_chla, model=model_chla)
        chla = chla.detach()

    if torch.count_nonzero(BBP700) == 0:
        BBP700 = get_output(sample=sample_prediction, model_day=model_day_BBP700, model_year=model_year_BBP700,
                            model_lat=model_lat_BBP700, model_lon=model_lon_BBP700, model=model_BBP700)
        BBP700 = BBP700.detach()
        BBP700 = BBP700 / 1000

    dict_sample = {name_float[0]: [year.item(), day_rad.item(), lat.item(), lon.item(),
                                   torch.squeeze(temp), torch.squeeze(psal), torch.squeeze(doxy),
                                   torch.squeeze(nitrate), torch.squeeze(chla), torch.squeeze(BBP700)]}

    dict_ds_enhanced = {**dict_ds_enhanced, **dict_sample}

pd_ds_enhanced = pd.DataFrame(dict_ds_enhanced,
                              index=['year', 'day_rad', 'lat', 'lon', 'temp', 'psal', 'doxy', 'nitrate', 'chla',
                                     'BBP700'])
pd_ds_enhanced.to_csv(os.getcwd() + f'/../ds/clustering/ds_sf_clustering_enhanced.csv')
