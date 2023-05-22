import os

import numpy as np
import pandas as pd
import netCDF4 as nc
import torch
from torch.utils.data import DataLoader

from discretization import *
from utils import upload_and_evaluate_model, get_output
from dataset_with_float_names import FloatDataset

dict_models = {
    "NITRATE": ["2023-04-04_", 50],
    "CHLA": ["2023-03-29", 150],
    "BBP700": ["2023-03-29", 100]
}

date_nitrate = dict_models["NITRATE"][0]
date_chla = dict_models["CHLA"][0]
date_BBP700 = dict_models["BBP700"][0]

dir_model_nitrate = os.getcwd() + f"/../results/NITRATE/{date_nitrate}/model"
dir_model_chla = os.getcwd() + f"/../results/CHLA/{date_chla}/model"
dir_model_BBP700 = os.getcwd() + f"/../results/BBP700/{date_BBP700}/model"

# Upload the input information
info_nitrate = pd.read_csv(os.getcwd() + f"/../results/NITRATE/{date_nitrate}/info.csv")
info_chla = pd.read_csv(os.getcwd() + f"/../results/CHLA/{date_chla}/info.csv")
info_BBP700 = pd.read_csv(os.getcwd() + f"/../results/BBP700/{date_BBP700}/info.csv")

# Upload and evaluate the model
model_day_nitrate, model_year_nitrate, model_lat_nitrate, model_lon_nitrate, model_nitrate = upload_and_evaluate_model(
    dir_model=dir_model_nitrate, info_model=info_nitrate, ep=dict_models["NITRATE"][1])
model_day_chla, model_year_chla, model_lat_chla, model_lon_chla, model_chla = upload_and_evaluate_model(
    dir_model=dir_model_chla, info_model=info_chla, ep=dict_models["CHLA"][1])
model_day_BBP700, model_year_BBP700, model_lat_BBP700, model_lon_BBP700, model_BBP700 = upload_and_evaluate_model(
    dir_model=dir_model_BBP700, info_model=info_BBP700, ep=dict_models["BBP700"][1])

# read the float number + sampling number from the dataset
path_df = f"/home/gpietropolli/Desktop/canyon-float/ds/clustering/ds_sf_clustering.csv"
dataset = FloatDataset(path_df)
my_ds = DataLoader(dataset, shuffle=True)

# information related to the pres of PPCon generated variables
pres_nitrate = np.arange(0, dict_max_pressure["NITRATE"], dict_interval["NITRATE"])
pres_chla = np.arange(0, dict_max_pressure["CHLA"], dict_interval["CHLA"])
pres_BBP700 = np.arange(0, dict_max_pressure["BBP700"], dict_interval["BBP700"])

qc = np.ones(200) * -9999

for sample in my_ds:
    year, day_rad, lat, lon, temp, psal, doxy, nitrate, chla, BBP700, name_float = sample
    sample_prediction = sample[:-1]

    name_file = name_float[0]
    main_dir = name_float[0][2:-4]
    print(f"generating profiles {name_file}")

    # open the netcfd file in the "ds/SUPERFLOAT" directory
    path_superfloat = f"/home/gpietropolli/Desktop/canyon-float/ds/SUPERFLOAT_PPCon/{main_dir}/{name_file}.nc"
    ds = nc.Dataset(path_superfloat, 'a')  # ds = nc.Dataset(path_superfloat)

    # create the dimension relative to the PPCon prediction
    if {'nNITRATE_PPCON', 'nCHLA_PPCON', 'nBBP700_PPCON'} & set(ds.dimensions.keys()):
        continue

    ds.createDimension('nNITRATE_PPCON', 200)
    ds.createDimension('nCHLA_PPCON', 200)
    ds.createDimension('nBBP700_PPCON', 200)

    # NITRATE routine
    nitrate_ppcon = get_output(sample=sample_prediction, model_day=model_day_nitrate, model_year=model_year_nitrate,
                               model_lat=model_lat_nitrate, model_lon=model_lon_nitrate, model=model_nitrate)
    nitrate_ppcon = nitrate_ppcon.detach()
    nitrate_ppcon = torch.squeeze(nitrate_ppcon)
    nitrate_ppcon = nitrate_ppcon.numpy()

    # check if the variable is contained in the ds - if not insert the variable generate also as "NITRATE"
    # if torch.count_nonzero(nitrate) == 0:
    if "NITRATE" not in ds.variables.keys():
        ds.createDimension('nNITRATE', 200)

        nc_nitrate = ds.createVariable('NITRATE', "f", ('nNITRATE',))
        nc_nitrate.units = "mmol/m3"
        nc_nitrate[:] = nitrate_ppcon

        nc_pres_nitrate = ds.createVariable('PRES_NITRATE', "f", ('nNITRATE',))
        nc_pres_nitrate.units = "m"
        nc_pres_nitrate[:] = pres_nitrate

        nc_qc_nitrate = ds.createVariable('QC_NITRATE', "f", ('nNITRATE',))
        nc_qc_nitrate[:] = qc

    # add the "NITRATE_PPCon" variable and "PRES" in any case
    nc_nitrate_ppcon = ds.createVariable('NITRATE_PPCON', "f", ('nNITRATE_PPCON',))
    nc_nitrate_ppcon.units = "mmol/m3"
    nc_nitrate_ppcon[:] = nitrate_ppcon

    nc_pres_nitrate_ppcon = ds.createVariable('PRES_NITRATE_PPCON', "f", ('nNITRATE_PPCON',))
    nc_pres_nitrate_ppcon.units = "m"
    nc_pres_nitrate_ppcon[:] = pres_nitrate

    nc_qc_nitrate_ppcon = ds.createVariable('QC_NITRATE_PPCON', "f", ('nNITRATE_PPCON',))
    nc_qc_nitrate_ppcon[:] = qc

    # CHLA routine
    chla_ppcon = get_output(sample=sample_prediction, model_day=model_day_chla, model_year=model_year_chla,
                            model_lat=model_lat_chla, model_lon=model_lon_chla, model=model_chla)
    chla_ppcon = chla_ppcon.detach()
    chla_ppcon = torch.squeeze(chla_ppcon)
    chla_ppcon = chla_ppcon.numpy()

    if "CHLA" not in ds.variables.keys():
        ds.createDimension('nCHLA', 200)

        nc_chla = ds.createVariable('CHLA', "f", ('nCHLA',))
        nc_chla.units = "mg/m3"
        nc_chla[:] = chla_ppcon

        nc_pres_chla = ds.createVariable('PRES_CHLA', "f", ('nCHLA',))
        nc_pres_chla.units = "m"
        nc_pres_chla[:] = pres_chla

        nc_qc_chla = ds.createVariable('QC_CHLA', "f", ('nCHLA',))
        nc_qc_chla[:] = qc

    # add the "CHLA_PPCon" variable and "PRES" in any case
    nc_chla_ppcon = ds.createVariable('CHLA_PPCON', "f", ('nCHLA_PPCON',))
    nc_chla_ppcon.units = "mg/m3"
    nc_chla_ppcon[:] = chla_ppcon

    nc_pres_chla_ppcon = ds.createVariable('PRES_CHLA_PPCON', "f", ('nCHLA_PPCON',))
    nc_pres_chla_ppcon.units = "m"
    nc_pres_chla_ppcon[:] = pres_chla

    nc_qc_chla_ppcon = ds.createVariable('QC_CHLA_PPCON', "f", ('nCHLA_PPCON',))
    nc_qc_chla_ppcon[:] = qc

    # BBP700 routine
    BBP700_ppcon = get_output(sample=sample_prediction, model_day=model_day_BBP700, model_year=model_year_BBP700,
                              model_lat=model_lat_BBP700, model_lon=model_lon_BBP700, model=model_BBP700)
    BBP700_ppcon = BBP700_ppcon.detach()
    BBP700_ppcon = BBP700_ppcon / 1000
    BBP700_ppcon = torch.squeeze(BBP700_ppcon)
    BBP700_ppcon = BBP700_ppcon.numpy()

    if "BBP700" not in ds.variables.keys():
        ds.createDimension('nBBP700', 200)

        nc_BBP700 = ds.createVariable('BBP700', "f", ('nBBP700',))
        nc_BBP700.units = "1/m"
        nc_BBP700[:] = BBP700_ppcon

        nc_pres_BBP700 = ds.createVariable('PRES_BBP700', "f", ('nBBP700',))
        nc_pres_BBP700.units = "m"
        nc_pres_BBP700[:] = pres_BBP700

        nc_qc_BBP700 = ds.createVariable('QC_BBP700', "f", ('nBBP700',))
        nc_qc_BBP700[:] = qc

    # add the "BBP700_PPCon" variable and "PRES" in any case
    nc_BBP700_ppcon = ds.createVariable('BBP700_PPCON', "f", ('nBBP700_PPCON',))
    nc_BBP700_ppcon.units = "1/m"
    nc_BBP700_ppcon[:] = BBP700_ppcon

    nc_pres_BBP700_ppcon = ds.createVariable('PRES_BBP700_PPCON', "f", ('nBBP700_PPCON',))
    nc_pres_BBP700_ppcon.units = "m"
    nc_pres_BBP700_ppcon[:] = pres_BBP700

    nc_qc_BBP700_ppcon = ds.createVariable('QC_BBP700_PPCON', "f", ('nBBP700_PPCON',))
    nc_qc_BBP700_ppcon[:] = qc

    # close the dataset
    ds.close()

    # save the new generated nc file in the enhanced directory
    # ds.to_netcdf(f"{path_saving_nc}/{name_file}.nc")
    # print(f"{name_file} saved successfully!")
