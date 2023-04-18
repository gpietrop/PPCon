import os
import random

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.basemap import Basemap

from discretization import dict_max_pressure
from dataset import FloatDataset
from utils import upload_and_evaluate_model, from_day_rad_to_day
from other_methods.suazade import get_suazade_profile
from other_methods.gloria import get_gloria_profile

sns.set_theme(context='notebook', style='whitegrid', palette='deep', font='sans-serif', font_scale=1,
              color_codes=True, rc=None)

dict_unit_measure = {"NITRATE": "mg/l",
                     "CHLA": "mg/m^3",
                     "BBP700": " "}
pal = sns.color_palette("magma")

dict_color = {'NWM': pal[0], 'SWM': pal[1], 'TIR': pal[3], 'ION': pal[4], 'LEV': pal[5]}


def count_samples(variable):
    path_ds = os.getcwd() + f"/../ds/{variable}/"
    ds_train = FloatDataset(path_ds + "float_ds_sf_train.csv")
    ds_test = FloatDataset(path_ds + "float_ds_sf_test.csv")
    ds_removed = FloatDataset(path_ds + "float_ds_sf_removed.csv")
    return len(ds_train), len(ds_test), len(ds_removed)


def get_reconstruction(variable, date_model, epoch_model, mode):
    # Upload the input ds
    path_float = f"/home/gpietropolli/Desktop/canyon-float/ds/{variable}/float_ds_sf_{mode}.csv"
    if mode == "all":
        path_float = f"/home/gpietropolli/Desktop/canyon-float/ds/{variable}/float_ds_sf.csv"
    dataset = FloatDataset(path_float)
    ds = DataLoader(dataset, shuffle=True)

    dir_model = os.getcwd() + f"/../results/{variable}/{date_model}/model"
    info = pd.read_csv(os.getcwd() + f"/../results/{variable}/{date_model}/info.csv")

    # Upload and evaluate the model
    model_day, model_year, model_lat, model_lon, model = upload_and_evaluate_model(dir_model=dir_model, info_model=info,
                                                                                   ep=epoch_model)

    lat_list = list()
    lon_list = list()
    day_rad_list = list()
    measured_var_list = list()
    generated_var_list = list()

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
        generated_var = generated_var.detach()  # torch.squeeze????

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


def get_reconstruction_comparison(season, date_model, epoch_model, mode):
    # Create savedir
    path_analysis = os.getcwd() + f"/../results/NITRATE/{date_model}/fig/comp"
    if not os.path.exists(path_analysis):
        os.mkdir(path_analysis)
    # Upload the input ds

    dict_season = {'W': [0, 91], 'SP': [92, 182], 'SU': [183, 273], 'A': [274, 365]}

    path_float = f"/home/gpietropolli/Desktop/canyon-float/ds/NITRATE/float_ds_sf_{mode}.csv"
    if mode == "all":
        path_float = f"/home/gpietropolli/Desktop/canyon-float/ds/NITRATE/float_ds_sf.csv"
    dataset = FloatDataset(path_float)
    ds = DataLoader(dataset, shuffle=True)

    dir_model = os.getcwd() + f"/../results/NITRATE/{date_model}/model"
    info = pd.read_csv(os.getcwd() + f"/../results/NITRATE/{date_model}/info.csv")

    # Upload and evaluate the model
    model_day, model_year, model_lat, model_lon, model = upload_and_evaluate_model(dir_model=dir_model, info_model=info,
                                                                                   ep=epoch_model)

    lat_list = list()
    lon_list = list()
    day_rad_list = list()
    measured_var_list = list()
    generated_var_list = list()
    suazade_var_list = list()
    gloria_var_list = list()
    number_seasonal_sample = 0

    sum_generated = torch.zeros(1, 1, 200)
    sum_measured = torch.zeros(1, 1, 200)
    sum_gloria = torch.zeros(1, 1, 200)
    sum_suazade = torch.zeros(200)

    for sample in ds:
        year, day_rad, lat, lon, temp, psal, doxy, measured_var = sample
        day_sample = from_day_rad_to_day(day_rad=day_rad)
        if season != "all" and not dict_season[season][0] <= day_sample <= dict_season[season][
            1] and random.random() < 0.2:
            continue
        number_seasonal_sample += 1
        generated_suazade_var = get_suazade_profile(year, day_rad, lat, lon, torch.squeeze(temp), torch.squeeze(psal),
                                                    torch.squeeze(doxy), torch.squeeze(measured_var))

        suazade_var_list.append(generated_suazade_var)
        sum_suazade += generated_suazade_var

        generated_gloria_var = get_gloria_profile(year, day_rad, lat, lon, torch.squeeze(temp), torch.squeeze(psal),
                                                  torch.squeeze(doxy), torch.squeeze(measured_var))
        gloria_var_list.append(generated_gloria_var)
        sum_gloria += generated_gloria_var

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
        generated_var = generated_var.detach()  # torch.squeeze????

        lat_list.append(lat.item())
        lon_list.append(lon.item())
        day_rad_list.append(day_rad.item())

        generated_var_list.append(generated_var)
        sum_generated += generated_var
        measured_var_list.append(measured_var)
        sum_measured += measured_var

        depth = np.linspace(0, dict_max_pressure["NITRATE"], len(torch.squeeze(generated_var)))

        cut_sup = 3
        cut_inf = 10

        plt.plot(torch.squeeze(measured_var)[cut_sup:-cut_inf], depth[cut_sup:-cut_inf], label="measured")
        plt.plot(torch.squeeze(generated_var)[cut_sup:-cut_inf], depth[cut_sup:-cut_inf], label="convolutional")
        plt.plot(torch.squeeze(generated_gloria_var)[cut_sup:-cut_inf], depth[cut_sup:-cut_inf],
                 label="canyon-med enhanced")
        plt.plot(generated_suazade_var[cut_sup:-cut_inf], depth[cut_sup:-cut_inf], label="canyon-med")
        plt.gca().invert_yaxis()

        plt.legend()
        plt.savefig(f"{path_analysis}/method_comparison_{round(lat.item(), 2)}_{round(lon.item(), 2)}.png")
        plt.close()

    generated_mean = sum_generated / number_seasonal_samples
    measured_mean = sum_measured / number_seasonal_samples
    gloria_mean = sum_gloria / number_seasonal_samples
    suazade_mean = sum_suazade / number_seasonal_samples

    max_pres = dict_max_pressure[variable]
    depth = np.linspace(0, max_pres, len(generated_profile.detach().numpy()))
    plt.plot(torch.squeeze(measured_mean), depth, label=f"measured")
    plt.plot(torch.squeeze(generated_mean), depth, label=f"convolutional")
    plt.plot(torch.squeeze(gloria_mean), depth, label=f"canyon-med enhanced")
    plt.plot(suazade_mean, depth, label=f"canyon-med")
    plt.gca().invert_yaxis()

    plt.legend()
    plt.savefig(f"{path_analysis}/method_comparison_{season}.png")
    plt.show()
    plt.close()

    return


def rmse(variable, date_model, epoch_model, mode):
    lat_list, lon_list, day_rad_list, generated_var_list, measured_var_list = get_reconstruction(variable, date_model,
                                                                                                 epoch_model, mode)
    my_loss = 0
    number_samples = len(generated_var_list)
    for index_sample in range(number_samples):
        loss_sample = mse_loss(generated_var_list[index_sample], measured_var_list[index_sample])
        my_loss += loss_sample
    my_loss = my_loss / number_samples
    return my_loss


def seasonal_rmse(season, variable, date_model, epoch_model, mode):
    dict_season = {'W': [0, 91], 'SP': [92, 182], 'SU': [183, 273], 'A': [274, 365]}
    lat_list, lon_list, day_rad_list, generated_var_list, measured_var_list = get_reconstruction(variable, date_model,
                                                                                                 epoch_model, mode)
    my_loss = 0
    number_samples = len(generated_var_list)
    number_seasonal_samples = 0
    for index_sample in range(number_samples):
        day_sample = from_day_rad_to_day(day_rad=day_rad_list[index_sample])
        if dict_season[season][0] <= day_sample <= dict_season[season][1]:
            number_seasonal_samples += 1
            loss_sample = mse_loss(generated_var_list[index_sample], measured_var_list[index_sample])
            my_loss += loss_sample
    my_loss = my_loss / number_seasonal_samples
    return my_loss, number_seasonal_samples


def seasonal_and_geographic_rmse(variable, date_model, epoch_model, mode):
    path_analysis = os.getcwd() + f"/../results/{variable}/{date_model}/"
    if not os.path.exists(path_analysis):
        os.mkdir(path_analysis)

    dict_ga = {'NWM': [[40, 45], [-2, 9.5]],
               'SWM': [[32, 40], [-2, 9.5]],
               'TIR': [[37, 45], [9.5, 16]],
               'ION': [[30, 37], [9.5, 22]],
               'LEV': [[30, 37], [22, 36]]}

    dict_season = {'W': [0, 91], 'SP': [92, 182], 'SU': [183, 273], 'A': [274, 365]}

    list_loss = [[0 for _ in range(len(list(dict_ga.keys())))] for _ in range(len(list(dict_season.keys())))]
    list_number_samples = [[0 for _ in range(len(list(dict_ga.keys())))] for _ in range(len(list(dict_season.keys())))]

    lat_list, lon_list, day_rad_list, generated_var_list, measured_var_list = get_reconstruction(variable, date_model,
                                                                                                 epoch_model, mode)

    number_samples = len(generated_var_list)
    for index_s in range(len(list(dict_season.keys()))):
        season = list(dict_season.keys())[index_s]
        for index_ga in range(len(list(dict_ga.keys()))):
            ga = list(dict_ga.keys())[index_ga]
            for i in range(number_samples):
                day_sample = from_day_rad_to_day(day_rad=day_rad_list[i])
                if dict_season[season][0] <= day_sample <= dict_season[season][1]:
                    if dict_ga[ga][0][0] <= lat_list[i] <= dict_ga[ga][0][1] and dict_ga[ga][1][0] <= lon_list[i] <= \
                            dict_ga[ga][1][1]:
                        loss_sample = mse_loss(generated_var_list[i], measured_var_list[i])
                        # print(list_loss)
                        list_loss[index_s][index_ga] += loss_sample
                        list_number_samples[index_s][index_ga] += 1

    list_loss = np.array(list_loss) / np.array(list_number_samples)
    for j in range(len(list(dict_season.keys()))):
        print(f"{list(dict_season.keys())[j]} losses: \t number samples")
        for k in range(len(list(dict_ga.keys()))):
            print(f"{list(dict_ga.keys())[k]} : {list_loss[j, k]} \t {list_number_samples[j][k]}")
    return


def geographic_rmse(lat_limits, lon_limits, variable, date_model, epoch_model, mode):
    lat_list, lon_list, day_rad_list, generated_var_list, measured_var_list = get_reconstruction(variable, date_model,
                                                                                                 epoch_model, mode)
    my_loss = 0
    number_samples = len(generated_var_list)
    number_geographic_samples = 0
    for i_sample in range(number_samples):
        if lat_limits[0] <= lat_list[i_sample] <= lat_limits[1] and lon_limits[0] <= lon_list[i_sample] <= lon_limits[
            1]:
            number_geographic_samples += 1
            loss_sample = mse_loss(generated_var_list[i_sample], measured_var_list[i_sample])
            my_loss += loss_sample
    my_loss = my_loss / number_geographic_samples
    return my_loss, number_geographic_samples


def seasonal_profile(season, variable, date_model, epoch_model, mode):
    dict_season = {'W': [0, 91], 'SP': [92, 182], 'SU': [183, 273], 'A': [274, 365]}
    lat_list, lon_list, day_rad_list, generated_var_list, measured_var_list = get_reconstruction(variable, date_model,
                                                                                                 epoch_model, mode)
    len_profile = int(generated_var_list[0].size(dim=0))

    generated_profile = torch.zeros(len_profile)
    measured_profile = torch.zeros(len_profile)
    number_samples = len(generated_var_list)
    number_seasonal_samples = 0
    for index_sample in range(number_samples):
        day_sample = from_day_rad_to_day(day_rad=day_rad_list[index_sample])
        if dict_season[season][0] <= day_sample <= dict_season[season][1]:
            number_seasonal_samples += 1
            generated_profile = generated_profile + generated_var_list[index_sample]
            measured_profile = measured_profile + measured_var_list[index_sample]

    generated_profile = generated_profile / number_seasonal_samples
    measured_profile = measured_profile / number_seasonal_samples

    max_pres = dict_max_pressure[variable]
    depth = np.linspace(0, max_pres, len(generated_profile.detach().numpy()))
    plt.plot(generated_profile.detach().numpy(), depth, label=f"generated {variable}")
    plt.plot(measured_profile.detach().numpy(), depth, label=f"measured {variable}")
    plt.gca().invert_yaxis()

    plt.legend()
    plt.show()
    plt.close()

    return


def seasonal_ga_profile(season, ga, lat_list, lon_list, day_rad_list, generated_var_list, measured_var_list):
    dict_ga = {'NWM': [[40, 45], [-2, 9.5]],
               'SWM': [[32, 40], [-2, 9.5]],
               'TIR': [[37, 45], [9.5, 16]],
               'ION': [[30, 37], [9.5, 22]],
               'LEV': [[30, 37], [22, 36]]}

    dict_season = {'W': [0, 91], 'SP': [92, 182], 'SU': [183, 273], 'A': [274, 365]}

    len_profile = int(generated_var_list[0].size(dim=0))

    generated_profile = torch.zeros(len_profile)
    measured_profile = torch.zeros(len_profile)
    number_samples = len(generated_var_list)
    number_seasonal_samples = 0
    for i in range(number_samples):
        day_sample = from_day_rad_to_day(day_rad=day_rad_list[i])
        if dict_season[season][0] <= day_sample <= dict_season[season][1]:
            if dict_ga[ga][0][0] <= lat_list[i] <= dict_ga[ga][0][1] and dict_ga[ga][1][0] <= lon_list[i] <= \
                    dict_ga[ga][1][1]:
                number_seasonal_samples += 1
                generated_profile = generated_profile + generated_var_list[i]
                measured_profile = measured_profile + measured_var_list[i]

    generated_profile = generated_profile / number_seasonal_samples
    measured_profile = measured_profile / number_seasonal_samples

    return generated_profile, measured_profile


def profile(variable, date_model, epoch_model, mode):
    path_analysis = os.getcwd() + f"/../results/{variable}/{date_model}/fig/"
    if not os.path.exists(path_analysis):
        os.mkdir(path_analysis)

    dict_ga = {'NWM': [[40, 45], [-2, 9.5]],
               'SWM': [[32, 40], [-2, 9.5]],
               'TIR': [[37, 45], [9.5, 16]],
               'ION': [[30, 37], [9.5, 22]],
               'LEV': [[30, 37], [22, 36]]}

    dict_season = {'W': [0, 91], 'SP': [92, 182], 'SU': [183, 273], 'A': [274, 365]}

    lat_list, lon_list, day_rad_list, generated_var_list, measured_var_list = get_reconstruction(variable, date_model,
                                                                                                 epoch_model, mode)

    fig, axs = plt.subplots(2, 2)

    for ga in list(dict_ga.keys()):
        generated, measured = seasonal_ga_profile("W", ga, lat_list, lon_list, day_rad_list, generated_var_list,
                                                  measured_var_list)
        max_pres = dict_max_pressure[variable]
        depth = np.linspace(0, max_pres, len(generated.detach().numpy()))

        axs[0, 0].plot(generated.detach().numpy(), depth,
                       linestyle="dashed", color=dict_color[ga])
        axs[0, 0].plot(measured.detach().numpy(), depth,
                       label=ga, linestyle="solid", color=dict_color[ga])
        axs[0, 0].invert_yaxis()

        generated, measured = seasonal_ga_profile("SP", ga, lat_list, lon_list, day_rad_list, generated_var_list,
                                                  measured_var_list)
        axs[0, 1].plot(generated.detach().numpy(), depth,
                       linestyle="dashed", color=dict_color[ga])
        axs[0, 1].plot(measured.detach().numpy(), depth,
                       label=ga, linestyle="solid", color=dict_color[ga])
        axs[0, 1].invert_yaxis()

        generated, measured = seasonal_ga_profile("SU", ga, lat_list, lon_list, day_rad_list, generated_var_list,
                                                  measured_var_list)
        axs[1, 0].plot(generated.detach().numpy(), depth,
                       linestyle="dashed", color=dict_color[ga])
        axs[1, 0].plot(measured.detach().numpy(), depth,
                       label=ga, linestyle="solid", color=dict_color[ga])
        axs[1, 0].invert_yaxis()

        generated, measured = seasonal_ga_profile("A", ga, lat_list, lon_list, day_rad_list, generated_var_list,
                                                  measured_var_list)
        axs[1, 1].plot(generated.detach().numpy(), depth,
                       linestyle="dashed", color=dict_color[ga])
        axs[1, 1].plot(measured.detach().numpy(), depth,
                       label=ga, linestyle="solid", color=dict_color[ga])
        axs[1, 1].invert_yaxis()

    axs[0, 0].set_title(list(dict_season.keys())[0])
    axs[0, 1].set_title(list(dict_season.keys())[1])
    axs[1, 0].set_title(list(dict_season.keys())[2])
    axs[1, 1].set_title(list(dict_season.keys())[3])

    for ax in axs.flat:
        ax.set(xlabel=f"{variable} ({dict_unit_measure[variable]})", ylabel='depth')
        # ax.set_xticks(range(1, len(list(dict_ga.keys())) + 1), list(dict_ga.keys()))

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize="5")

    plt.savefig(f"{path_analysis}profile_comparison_{mode}_{epoch_model}.png")

    # plt.show()
    plt.close()

    return


def profile_error(variable, date_model, epoch_model, mode):
    path_analysis = os.getcwd() + f"/../results/{variable}/{date_model}/fig/"
    if not os.path.exists(path_analysis):
        os.mkdir(path_analysis)

    dict_ga = {'NWM': [[40, 45], [-2, 9.5]],
               'SWM': [[32, 40], [-2, 9.5]],
               'TIR': [[37, 45], [9.5, 16]],
               'ION': [[30, 37], [9.5, 22]],
               'LEV': [[30, 37], [22, 36]]}

    dict_season = {'W': [0, 91], 'SP': [92, 182], 'SU': [183, 273], 'A': [274, 365]}

    lat_list, lon_list, day_rad_list, generated_var_list, measured_var_list = get_reconstruction(variable, date_model,
                                                                                                 epoch_model, mode)

    fig, axs = plt.subplots(2, 2)

    for ga in list(dict_ga.keys()):
        generated, measured = seasonal_ga_profile("W", ga, lat_list, lon_list, day_rad_list, generated_var_list,
                                                  measured_var_list)
        max_pres = dict_max_pressure[variable]
        depth = np.linspace(0, max_pres, len(generated.detach().numpy()))

        print(measured.detach().numpy())
        print(generated.detach().numpy())

        axs[0, 0].plot(np.abs(measured.detach().numpy()) -
                       np.abs(generated.detach().numpy()), depth,
                       label=ga, linestyle="solid", color=dict_color[ga])
        axs[0, 0].invert_yaxis()

        generated, measured = seasonal_ga_profile("SP", ga, lat_list, lon_list, day_rad_list, generated_var_list,
                                                  measured_var_list)
        axs[0, 1].plot(np.abs(measured.detach().numpy()) -
                       np.abs(generated.detach().numpy()), depth,
                       label=ga, linestyle="solid", color=dict_color[ga])
        axs[0, 1].invert_yaxis()

        generated, measured = seasonal_ga_profile("SU", ga, lat_list, lon_list, day_rad_list, generated_var_list,
                                                  measured_var_list)
        axs[1, 0].plot(np.abs(measured.detach().numpy()) -
                       np.abs(generated.detach().numpy()), depth,
                       label=ga, linestyle="solid", color=dict_color[ga])
        axs[1, 0].invert_yaxis()

        generated, measured = seasonal_ga_profile("A", ga, lat_list, lon_list, day_rad_list, generated_var_list,
                                                  measured_var_list)
        axs[1, 1].plot(np.abs(measured.detach().numpy()) -
                       np.abs(generated.detach().numpy()), depth,
                       label=ga, linestyle="solid", color=dict_color[ga])
        axs[1, 1].invert_yaxis()

    axs[0, 0].set_title(list(dict_season.keys())[0])
    axs[0, 1].set_title(list(dict_season.keys())[1])
    axs[1, 0].set_title(list(dict_season.keys())[2])
    axs[1, 1].set_title(list(dict_season.keys())[3])

    for ax in axs.flat:
        ax.set(xlabel=f"{variable} ({dict_unit_measure[variable]})", ylabel='depth')
        # ax.set_xticks(range(1, len(list(dict_ga.keys())) + 1), list(dict_ga.keys()))

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize="5")

    plt.savefig(f"{path_analysis}profile_error_{mode}_{epoch_model}.png")

    # plt.show()
    plt.close()

    return


def seasonal_ga_variance(season, ga, lat_list, lon_list, day_rad_list, generated_var_list, measured_var_list):
    dict_ga = {'NWM': [[40, 45], [-2, 9.5]],
               'SWM': [[32, 40], [-2, 9.5]],
               'TIR': [[37, 45], [9.5, 16]],
               'ION': [[30, 37], [9.5, 22]],
               'LEV': [[30, 37], [22, 36]]}

    dict_season = {'W': [0, 91], 'SP': [92, 182], 'SU': [183, 273], 'A': [274, 365]}

    len_profile = int(generated_var_list[0].size(dim=0))

    generated_profile = []
    # measured_profile = []
    number_samples = len(generated_var_list)
    number_seasonal_samples = 0
    for i in range(number_samples):
        day_sample = from_day_rad_to_day(day_rad=day_rad_list[i])
        if dict_season[season][0] <= day_sample <= dict_season[season][1]:
            if dict_ga[ga][0][0] <= lat_list[i] <= dict_ga[ga][0][1] and dict_ga[ga][1][0] <= lon_list[i] <= \
                    dict_ga[ga][1][1]:
                number_seasonal_samples += 1
                generated_profile.append(generated_var_list[i])
                # measured_profile.append(measured_var_list[i])

    std_reconstruction = torch.zeros(len_profile)
    for k in range(len_profile):
        reconstruction_depth = []
        for prof in generated_profile:
            reconstruction_depth.append(prof[k].item())
        # print(reconstruction_depth)
        std_reconstruction[k] = np.std(reconstruction_depth)

    return std_reconstruction


def profile_variance(variable, date_model, epoch_model, mode):
    path_analysis = os.getcwd() + f"/../results/{variable}/{date_model}/fig/"
    if not os.path.exists(path_analysis):
        os.mkdir(path_analysis)

    dict_ga = {'NWM': [[40, 45], [-2, 9.5]],
               'SWM': [[32, 40], [-2, 9.5]],
               'TIR': [[37, 45], [9.5, 16]],
               'ION': [[30, 37], [9.5, 22]],
               'LEV': [[30, 37], [22, 36]]}

    dict_season = {'W': [0, 91], 'SP': [92, 182], 'SU': [183, 273], 'A': [274, 365]}

    lat_list, lon_list, day_rad_list, generated_var_list, measured_var_list = get_reconstruction(variable, date_model,
                                                                                                 epoch_model, mode)

    fig, axs = plt.subplots(2, 2)

    for ga in list(dict_ga.keys()):
        variance = seasonal_ga_variance("W", ga, lat_list, lon_list, day_rad_list, generated_var_list,
                                        measured_var_list)
        max_pres = dict_max_pressure[variable]
        depth = np.linspace(0, max_pres, len(variance.detach().numpy()))

        axs[0, 0].plot(variance.detach().numpy(), depth, color=dict_color[ga], label=ga)
        axs[0, 0].invert_yaxis()

        variance = seasonal_ga_variance("SP", ga, lat_list, lon_list, day_rad_list, generated_var_list,
                                        measured_var_list)
        axs[0, 1].plot(variance.detach().numpy(), depth, color=dict_color[ga], label=ga)
        axs[0, 1].invert_yaxis()

        variance = seasonal_ga_variance("SU", ga, lat_list, lon_list, day_rad_list, generated_var_list,
                                        measured_var_list)
        axs[1, 0].plot(variance.detach().numpy(), depth, color=dict_color[ga], label=ga)
        axs[1, 0].invert_yaxis()

        variance = seasonal_ga_variance("A", ga, lat_list, lon_list, day_rad_list, generated_var_list,
                                        measured_var_list)
        axs[1, 1].plot(variance.detach().numpy(), depth, color=dict_color[ga], label=ga)
        axs[1, 1].invert_yaxis()

    axs[0, 0].set_title(list(dict_season.keys())[0])
    axs[0, 1].set_title(list(dict_season.keys())[1])
    axs[1, 0].set_title(list(dict_season.keys())[2])
    axs[1, 1].set_title(list(dict_season.keys())[3])

    for ax in axs.flat:
        ax.set(xlabel=f"{variable} ({dict_unit_measure[variable]})", ylabel='depth')
        # ax.set_xticks(range(1, len(list(dict_ga.keys())) + 1), list(dict_ga.keys()))

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize="5")

    plt.savefig(f"{path_analysis}variance_comparison_{mode}_{epoch_model}.png")

    # plt.show()
    plt.close()

    return


def seasonal_bp(variable, date_model, epoch_model, mode):
    path_analysis = os.getcwd() + f"/../results/{variable}/{date_model}/"
    if not os.path.exists(path_analysis):
        os.mkdir(path_analysis)

    dict_season = {'W': [0, 91], 'SP': [92, 182], 'SU': [183, 273], 'A': [274, 365]}
    list_loss_for_season = [[] for i in range(4)]

    lat_list, lon_list, day_rad_list, generated_var_list, measured_var_list = get_reconstruction(variable, date_model,
                                                                                                 epoch_model, mode)
    number_samples = len(generated_var_list)
    for index_season in range(4):
        season = list(dict_season.keys())[index_season]
        for index_sample in range(number_samples):
            day_sample = from_day_rad_to_day(day_rad=day_rad_list[index_sample])
            if dict_season[season][0] <= day_sample <= dict_season[season][1]:
                loss_sample = mse_loss(generated_var_list[index_sample], measured_var_list[index_sample])
                list_loss_for_season[index_season].append(loss_sample)

    sns.boxplot(data=list_loss_for_season,
                showfliers=False)
    plt.xticks(range(4), list(dict_season.keys()))
    plt.show()
    plt.close()

    return


def geographic_bp(variable, date_model, epoch_model, mode):
    path_analysis = os.getcwd() + f"/../results/{variable}/{date_model}/"
    if not os.path.exists(path_analysis):
        os.mkdir(path_analysis)

    dict_ga = {'NWM': [[40, 45], [-2, 9.5]],
               'SWM': [[32, 40], [-2, 9.5]],
               'TIR': [[37, 45], [9.5, 16]],
               'ION': [[30, 37], [9.5, 22]],
               'LEV': [[30, 37], [22, 36]]}
    list_loss_ga = [[] for _ in range(len(list(dict_ga.keys())))]

    lat_list, lon_list, day_rad_list, generated_var_list, measured_var_list = get_reconstruction(variable, date_model,
                                                                                                 epoch_model, mode)
    number_samples = len(generated_var_list)
    for index_ga in range(len(list(dict_ga.keys()))):
        ga = list(dict_ga.keys())[index_ga]
        for i in range(number_samples):
            if dict_ga[ga][0][0] <= lat_list[i] <= dict_ga[ga][0][1] and dict_ga[ga][1][0] <= lon_list[i] <= \
                    dict_ga[ga][1][1]:
                loss_sample = mse_loss(generated_var_list[i], measured_var_list[i])
                list_loss_ga[index_ga].append(loss_sample)

    sns.boxplot(data=list_loss_ga,
                showfliers=False)
    plt.xticks(range(len(list(dict_ga.keys()))), list(dict_ga.keys()))
    plt.show()
    plt.close()

    return


def seasonal_and_geographic_bp(variable, date_model, epoch_model, mode):
    path_analysis = os.getcwd() + f"/../results/{variable}/{date_model}/fig/"
    if not os.path.exists(path_analysis):
        os.mkdir(path_analysis)

    dict_ga = {'NWM': [[40, 45], [-2, 9.5]],
               'SWM': [[32, 40], [-2, 9.5]],
               'TIR': [[37, 45], [9.5, 16]],
               'ION': [[30, 37], [9.5, 22]],
               'LEV': [[30, 37], [22, 36]]}

    dict_season = {'W': [0, 91], 'SP': [92, 182], 'SU': [183, 273], 'A': [274, 365]}

    list_loss = [[[] for _ in range(len(list(dict_ga.keys())))] for _ in range(len(list(dict_season.keys())))]

    lat_list, lon_list, day_rad_list, generated_var_list, measured_var_list = get_reconstruction(variable, date_model,
                                                                                                 epoch_model, mode)
    number_samples = len(generated_var_list)
    for index_s in range(len(list(dict_season.keys()))):
        season = list(dict_season.keys())[index_s]
        for index_ga in range(len(list(dict_ga.keys()))):
            ga = list(dict_ga.keys())[index_ga]
            for i in range(number_samples):
                day_sample = from_day_rad_to_day(day_rad=day_rad_list[i])
                if dict_season[season][0] <= day_sample <= dict_season[season][1]:
                    if dict_ga[ga][0][0] <= lat_list[i] <= dict_ga[ga][0][1] and dict_ga[ga][1][0] <= lon_list[i] <= \
                            dict_ga[ga][1][1]:
                        loss_sample = mse_loss(generated_var_list[i], measured_var_list[i])
                        list_loss[index_s][index_ga].append(loss_sample)

    fig, axs = plt.subplots(2, 2)

    sns.boxplot(list_loss[0],
                palette="magma",
                ax=axs[0, 0])
    axs[0, 0].set_title(list(dict_season.keys())[0])

    sns.boxplot(list_loss[1],
                palette="magma",
                ax=axs[0, 1])
    axs[0, 1].set_title(list(dict_season.keys())[1])

    sns.boxplot(list_loss[2],
                palette="magma",
                ax=axs[1, 0])
    axs[1, 0].set_title(list(dict_season.keys())[2])

    sns.boxplot(list_loss[3],
                palette="magma",
                ax=axs[1, 1])
    axs[1, 1].set_title(list(dict_season.keys())[3])

    for ax in axs.flat:
        ax.set(xlabel=f"{variable} ({dict_unit_measure[variable]})", ylabel='fitness')
        if variable == "NITRATE":
            ax.set_ylim([0.0, 1])
        if variable == "CHLA":
            ax.set_ylim([0.0, 0.08])
        if variable == "BBP700":
            ax.set_ylim([0.0, 0.00000015])
        ax.set_xticks(range(0, len(list(dict_ga.keys()))), list(dict_ga.keys()))

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    plt.savefig(f"{path_analysis}bp_comparison_{mode}_{epoch_model}.png")

    # plt.show()
    plt.close()

    return


def plot_ga_med(lat_values, lon_values, clusters, lat_limits_l=30, lat_limits_u=47, lon_limits_l=-3, lon_limits_u=37):
    lon_limits = [lon_limits_l, lon_limits_u]
    lat_limits = [lat_limits_l, lat_limits_u]
    fig = plt.figure(figsize=(8, 8))
    mediterranean_map = Basemap(llcrnrlon=lon_limits[0],
                                llcrnrlat=lat_limits[0],
                                urcrnrlon=lon_limits[1],
                                urcrnrlat=lat_limits[1],
                                resolution='h')
    mediterranean_map.drawmapboundary(fill_color='aqua')
    mediterranean_map.fillcontinents(color='coral', lake_color='aqua')
    mediterranean_map.drawcoastlines()

    plt.scatter(lon_values, lat_values, c=clusters)
    plt.ylim(lat_limits)
    plt.xlim(lon_limits)

    plt.show()
    plt.close()


def plot_seasonal_ga_med(season, variable, date_model, epoch_model, mode):
    path_analysis = os.getcwd() + f"/../results/{variable}/{date_model}/"
    if not os.path.exists(path_analysis):
        os.mkdir(path_analysis)

    dict_season = {'W': [0, 91], 'SP': [92, 182], 'SU': [183, 273], 'A': [274, 365]}

    dict_ga = {'NWM': [[40, 45], [-2, 9.5]],
               'SWM': [[32, 40], [-2, 9.5]],
               'TIR': [[37, 45], [9.5, 16]],
               'ION': [[30, 37], [9.5, 22]],
               'LEV': [[30, 37], [22, 36]]}

    dict_c = {'NWM': 0, 'SWM': 1, 'TIR': 2, 'ION': 3, 'LEV': 4}

    lat_list_season = []
    lon_list_season = []
    c_list_season = []

    lat_list, lon_list, day_rad_list, generated_var_list, measured_var_list = get_reconstruction(variable, date_model,
                                                                                                 epoch_model, mode)
    number_samples = len(generated_var_list)

    for i in range(number_samples):
        day_sample = from_day_rad_to_day(day_rad=day_rad_list[i])
        if dict_season[season][0] <= day_sample <= dict_season[season][1]:

            for j in range(len(list(dict_ga.keys()))):
                key_ga = list(dict_ga.keys())[j]
                ga = dict_ga[key_ga]
                if ga[0][0] <= lat_list[i] <= ga[0][1] and ga[1][0] <= lon_list[i] <= ga[1][1]:
                    lat_list_season.append(lat_list[i])
                    lon_list_season.append(lon_list[i])
                    c_list_season.append(dict_c[key_ga])

    plot_ga_med(lat_list_season, lon_list_season, c_list_season)


var = "CHLA"
date = "2023-03-29"
# epoch = 150  # 150

N_lat = [39, 47]
S_lat = [30, 39]
whole_lat = [30, 47]
W_lon = [-3, 17]
E_lon = [17, 37]
whole_lon = [-3, 37]

"""
seasonal_and_geographic_rmse(var, date, epoch_model=150, mode="train")
seasonal_and_geographic_rmse(var, date, epoch_model=150, mode="test")
# seasonal_and_geographic_rmse(var, date, epoch, "test")
for epoch in [100, 150, 200]:
    # plot_seasonal_ga_med("SU", var, date, epoch, "test")
    seasonal_and_geographic_bp(var, date, epoch, "train")
    seasonal_and_geographic_bp(var, date, epoch, "test")
    profile(var, date, epoch, "train")
    profile(var, date, epoch, "test")
"""
