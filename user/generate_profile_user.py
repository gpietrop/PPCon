import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from user.utils_user import dict_max_pressure, dict_var_name, dict_unit_measure
from user.utils_user import get_reconstruction_user, moving_average

data = (
    torch.tensor([2021.0, 2022.0]),  # year
    torch.tensor([0.5, 0.6]),  # day_rad
    torch.tensor([35.0, 36.0]),  # lat
    torch.tensor([-120.0, -121.0]),  # lon
    torch.ones(2, 200),  # temp
    torch.ones(2, 200),  # psal
    torch.ones(2, 200),  # doxy
    torch.ones(2, 200)  # measured_var
)

dict_models = {
    "NITRATE": ["2023-12-16", 100],
    "CHLA": ["2023-12-17", 150],
    "BBP700": ["2023-12-15", 125]
}

variable = 'CHLA'

lat_list, lon_list, day_rad_list, generated_var_list, measured_var_list = get_reconstruction_user(
    variable=variable,
    date_model=dict_models[variable][0],
    epoch_model=dict_models[variable][1],
    mode='all',
    data=data
)

number_samples = len(generated_var_list)

for index_sample in range(number_samples):
    lat = lat_list[index_sample]
    lon = lon_list[index_sample]
    generated_profile = generated_var_list[index_sample]
    measured_profile = measured_var_list[index_sample]

    max_pres = dict_max_pressure[variable]
    depth = np.linspace(0, max_pres, len(generated_profile.detach().numpy()))
    plt.figure(figsize=(6, 7))

    measured_profile = moving_average(measured_profile.detach().numpy(), 3)
    generated_profile = moving_average(generated_profile.detach().numpy(), 3)
    # generated_profile = moving_average(generated_profile.detach().numpy(), 3)

    plt.plot(measured_profile, depth, lw=3, color="#2CA02C", label=f"Measured")
    plt.plot(generated_profile, depth, lw=3, linestyle=(0, (3, 1, 1, 1)), color="#1F77B4", label=f"PPCon")
    plt.gca().invert_yaxis()

    plt.xlabel(f"{dict_var_name[variable]} [{dict_unit_measure[variable]}]")
    plt.ylabel(r"Depth [$m$]")

    if variable == "BBP700":
        ax = plt.gca()
        x_labels = ax.get_xticks()
        ax.set_xticklabels(['{:,.0e}'.format(x) for x in x_labels])

    plt.legend()
    plt.tight_layout()

    plt.legend()
    plt.show()
    plt.close()

