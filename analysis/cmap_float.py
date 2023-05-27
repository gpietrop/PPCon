import os

import matplotlib
import numpy as np
import netCDF4 as nc

import matplotlib.pyplot as plt

from discretization import *
from make_ds.make_superfloat_ds import discretize

# I need as input the folder which contains float measurements and then take all the float vector one after each other

dir_list = os.listdir(f"/Users/admin/Desktop/ppcon/ds/SUPERFLOAT_PPCon/")
for var in ["NITRATE", "CHLA", "BBP700"]:
    for folder_name in dir_list:  # ["6901773"]:
        if folder_name[0] == "F" or folder_name[0] == ".":
            continue

        if var == "CHLA":
            maxmax = 0.5
            minmin = 0.0
        if var == "BBP700":
            maxmax = 0.004
            minmin = 0.000

        folder_path = f"/Users/admin/Desktop/ppcon/ds/SUPERFLOAT_PPCon/{folder_name}"

        # get ordere list of measurements
        files = os.listdir(folder_path)
        files.sort()

        # initialize the matrix for the cmap function
        matrix_measured = np.zeros((200, len(files)))
        matrix_generated = np.zeros((200, len(files)))

        # for each of the measurements get the original measurements and the prediction
        counter_discarded = 0
        flag_print = 0
        x_ticks = []  # date of sampling corresponding to the sampling value
        for index in range(len(files)):

            nc_file = files[index]
            nc_path = folder_path + "/" + nc_file
            try:
                ds = nc.Dataset(nc_path, "r")
            except Exception as error:
                continue

            date = ds["REFERENCE_DATE_TIME"][:]
            date = [str(np.ma.getdata(date)[k])[2] for k in range(len(date))]
            date = date[0] + date[1] + date[2] + date[3] + " " + date[4] + date[5]
            x_ticks.append(date)

            if not flag_print:
                lat = float(ds["LATITUDE"][:])
                lon = float(ds["LONGITUDE"][:])
                flag_print = 1

            if "DOXY" not in ds.variables.keys():
                counter_discarded -= 1
                matrix_measured[:, index] = -999 * np.array(200)
                matrix_generated[:, index] = -999 * np.array(200)
                continue

            if var not in ds.variables.keys() or f"{var}_PPCON" not in ds.variables.keys() or len(
                    ds[f"PRES_{var}"][:].data) == 200:
                matrix_measured[:, index] = -999 * np.array(200)
                matrix_generated[:, index] = -999 * np.array(200)
                counter_discarded -= 1
                continue

            var_measured = ds[var][:].data
            pres_var_measured = ds[f"PRES_{var}"][:].data

            var_measured_interpolated = discretize(pres_var_measured, var_measured, dict_max_pressure[var],
                                                   dict_interval[var])

            # matrix_measured[:, index + counter_discarded] = var_measured_interpolated
            matrix_measured[:, index] = var_measured_interpolated

            var_generated = ds[f"{var}_PPCON"][:].data
            pres_var_generated = ds[f"PRES_{var}_PPCON"][:].data

            # matrix_generated[:, index + counter_discarded] = var_generated
            matrix_generated[:, index] = var_generated

            # plt.plot(var_measured_interpolated, label="measured")
            # plt.plot(var_generated)
            # plt.legend()
            # plt.show()

        # mae = mean_absolute_error(matrix_generated, matrix_measured)
        # print(mae)

        # matrix_measured = matrix_measured[:, 1:counter_discarded]
        # matrix_generated = matrix_generated[:, 1:counter_discarded]

        # find minimum of minima & maximum of maxima
        try:
            if var == "NITRATE":
                minmin = 0
                max = np.max([np.max(matrix_measured), np.max(matrix_generated), 0])
                maxmax = 8
        except Exception as error:
            continue

        if np.max(matrix_measured) <= 0:
            continue

        x_number_ticks = np.arange(0, index)
        fig, axs = plt.subplots(2, figsize=(6, 5))

        cmap = matplotlib.cm.get_cmap('viridis').copy()
        cmap.set_under('white')

        im1 = axs[0].imshow(matrix_measured, vmin=minmin, vmax=maxmax,
                            cmap=cmap,
                            aspect='auto')  # , interpolation="nearest")
        axs[0].set_title("Measured")
        axs[0].set_xticks([])
        axs[0].set_yticks(np.arange(0, 200)[::50], np.arange(0, dict_max_pressure[var], dict_interval[var])[::50])

        im2 = axs[1].imshow(matrix_generated, vmin=minmin, vmax=maxmax,
                            cmap=cmap,
                            aspect='auto')  # , interpolation="bilinear")
        axs[1].set_title("PPCon prediction")
        axs[1].set_xticks(x_number_ticks[::25], x_ticks[::25], rotation=45)
        axs[1].tick_params(axis='x', labelsize=7)
        axs[1].set_yticks(np.arange(0, 200)[::50], np.arange(0, dict_max_pressure[var], dict_interval[var])[::50])

        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
        fig.colorbar(im2, cax=cbar_ax)

        fig.suptitle(f"lat: {round(lat, 2)}   lon: {round(lon, 2)}")

        # plt.colorbar()
        plt.savefig(f"/Users/admin/Desktop/ppcon/results/cmap/{var}/{folder_name}.png")

        plt.show()
        plt.close()
