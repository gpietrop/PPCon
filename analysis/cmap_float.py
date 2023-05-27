import os

import numpy as np
import netCDF4 as nc

import matplotlib.pyplot as plt

from discretization import *
from make_ds.make_superfloat_ds import discretize

# I need as input the folder which contains float measurements and then take all the float vector one after each other

dir_list = os.listdir(f"/Users/admin/Desktop/ppcon/ds/SUPERFLOAT_PPCon/")
for var in ["NITRATE", "CHLA", "BBP700"]:
    for folder_name in dir_list: # ["6901773"]:
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
        for index in range(len(files)):
            nc_file = files[index]
            nc_path = folder_path + "/" + nc_file
            try:
                ds = nc.Dataset(nc_path, "r")
            except Exception as error:
                continue

            if not flag_print:
                lat = float(ds["LATITUDE"][:])
                lon = float(ds["LONGITUDE"][:])
                # print(f"latitude: {lat}")
                # print(f"longitude: {lon}")
                flag_print = 1

            if "DOXY" not in ds.variables.keys():
                counter_discarded -= 1
                continue
            if var not in ds.variables.keys() or f"{var}_PPCON" not in ds.variables.keys() or len(
                    ds[f"PRES_{var}"][:].data) == 200:
                # print("float discarded")
                counter_discarded -= 1
                continue

            var_measured = ds[var][:].data
            pres_var_measured = ds[f"PRES_{var}"][:].data

            var_measured_interpolated = discretize(pres_var_measured, var_measured, dict_max_pressure[var],
                                                   dict_interval[var])
            matrix_measured[:, index + counter_discarded] = var_measured_interpolated
            matrix_measured[:, index] = var_measured_interpolated

            var_generated = ds[f"{var}_PPCON"][:].data
            pres_var_generated = ds[f"PRES_{var}_PPCON"][:].data
            matrix_generated[:, index + counter_discarded] = var_generated
            matrix_generated[:, index] = var_generated

            # plt.plot(var_measured_interpolated, label="measured")
            # plt.plot(var_generated)
            # plt.legend()
            # plt.show()

        # mae = mean_absolute_error(matrix_generated, matrix_measured)
        # print(mae)

        matrix_measured = matrix_measured[:, 1:counter_discarded]
        matrix_generated = matrix_generated[:, 1:counter_discarded]

        # find minimum of minima & maximum of maxima
        try:
            if var == "NITRATE":
                minmin = np.min([np.min(matrix_measured), np.min(matrix_generated)])
                maxmax = np.max([np.max(matrix_measured), np.max(matrix_generated)])
        except Exception as error:
            continue

        fig, axs = plt.subplots(2, figsize=(6, 5))

        im1 = axs[0].imshow(matrix_measured, vmin=minmin, vmax=maxmax, aspect='auto')  # , interpolation="nearest")
        axs[0].set_title("measured")

        im2 = axs[1].imshow(matrix_generated, vmin=minmin, vmax=maxmax, aspect='auto')  # , interpolation="bilinear")
        # axs[1].set_title("generated")

        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
        fig.colorbar(im2, cax=cbar_ax)

        fig.suptitle(f"lat: {round(lat, 2)}   lon: {round(lon, 2)}")

        # plt.colorbar()
        plt.savefig(f"/Users/admin/Desktop/ppcon/results/cmap/{var}/{folder_name}.png")

        # plt.show()
        plt.close()
