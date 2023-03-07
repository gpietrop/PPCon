import os

import torch
import numpy as np
import pandas as pd
import netCDF4 as nc

from discretization import dict_max_pressure, dict_interval


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def discretize(pres, var, max_pres, interval):
    """
    function that take as input a profile, the corresponding pres and create a tensor
    the interval input represents the discretization scale
    the max_pres input represents the maximum pressure considered
    """
    size = int(max_pres / interval)
    discretization_pres = np.arange(0, max_pres, interval)

    out = torch.zeros(size)

    for i in range(size):
        pressure_discretize = discretization_pres[i]
        idx = find_nearest(pres, pressure_discretize)
        if idx is None:
            return None
        out[i] = torch.from_numpy(np.asarray(var[idx]))

    return out


def read_date_time(date_time):
    year = int(date_time[0:4])
    month = int(date_time[4:6])
    month = month - 1
    day = int(date_time[6:8])

    day_total = month + day
    day_rad = day_total * 2 * np.pi / 365

    return year, day_rad


def make_dict_single_float(path, date_time, variable):
    if not os.path.exists(path):
        return None

    year, day_rad = read_date_time(date_time)

    try:
        ds = nc.Dataset(path)  # Select sample
    except Exception as error:
        print('Caught this error: ' + repr(error))
        return dict()

    lat = ds["LATITUDE"][:].data[0]
    lon = ds["LONGITUDE"][:].data[0]
    # pres_df = ds["PRES"][:].data[:]
    psal_df = ds["PSAL"][:].data[:]
    pres_psal_df = ds["PRES_PSAL"][:].data[:]

    temp_df = ds["TEMP"][:].data[:]
    pres_temp_df = ds["PRES_TEMP"][:].data[:]

    if "DOXY" not in ds.variables.keys():
        return dict()
    doxy_df = ds["DOXY"][:].data[:]
    pres_doxy_df = ds["PRES_DOXY"][:].data[:]

    if variable not in ds.variables.keys():
        return dict()
    variable_df = ds[f"{variable}"][:].data[:]
    pres_variable_df = ds[f"PRES_{variable}"][:].data[:]

    max_pres = dict_max_pressure[variable]
    interval = dict_interval[variable]

    temp = discretize(pres_temp_df, temp_df, max_pres, interval)
    psal = discretize(pres_psal_df, psal_df, max_pres, interval)
    doxy = discretize(pres_doxy_df, doxy_df, max_pres, interval)

    variable = discretize(pres_variable_df, variable_df, max_pres, interval)
    name_float = path[8:-3]
    if temp is None or psal is None or doxy is None or variable is None:
        return dict()
    dict_float = {name_float: [year, day_rad, lat, lon, temp, psal, doxy, variable]}
    return dict_float


def make_dataset(path_float_index, variable):
    name_list = pd.read_csv(path_float_index, header=None).to_numpy()[:, 0].tolist()
    datetime_list = pd.read_csv(path_float_index, header=None).to_numpy()[:, 3].tolist()

    dict_ds = dict()

    for i in range(np.size(name_list)):
        path = os.getcwd() + "/ds/SUPERFLOAT/" + name_list[i]
        if not os.path.exists(path):
            continue
        date_time = datetime_list[i]
        dict_single_float = make_dict_single_float(path, date_time, variable)
        dict_ds = {**dict_ds, **dict_single_float}

    return dict_ds


def make_pandas_df(path_float_index, variable):
    dict_ds = make_dataset(path_float_index, variable)
    pd_ds = pd.DataFrame(dict_ds, index=['year', 'day_rad', 'lat', 'lon', 'temp', 'psal', 'doxy', variable])

    pd_ds.to_csv(os.getcwd() + f'/ds/{variable}/float_ds_sf.csv')
    return


def make_toy_dataset(path_float_index, variable):
    name_list = pd.read_csv(path_float_index, header=None).to_numpy()[:, 0].tolist()
    datetime_list = pd.read_csv(path_float_index, header=None).to_numpy()[:, 3].tolist()

    dict_ds = dict()

    for i in range(int(np.size(name_list)/20)):
        path = os.getcwd() + "/ds/SUPERFLOAT/" + name_list[i]
        if not os.path.exists(path):
            continue
        date_time = datetime_list[i]
        dict_single_float = make_dict_single_float(path, date_time, variable)
        dict_ds = {**dict_ds, **dict_single_float}

    return dict_ds


def make_pandas_toy_df(path_float_index, variable):
    dict_ds = make_toy_dataset(path_float_index, variable)
    pd_ds = pd.DataFrame(dict_ds, index=['year', 'day_rad', 'lat', 'lon', 'temp', 'psal', 'doxy', variable])

    pd_ds.to_csv(os.getcwd() + f'/ds/{variable}/toy_ds_sf.csv')
    return
