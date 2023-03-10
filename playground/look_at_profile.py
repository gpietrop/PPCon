import os

import netCDF4 as nc
import pandas as pd

from make_ds.make_coriolis_ds import discretize

path = os.getcwd() + '/../ds/SUPERFLOAT/Float_Index.txt'
name_list = pd.read_csv(path, header=None).to_numpy()[:, 0].tolist()

i = 1000
flag = 0
var_name = "CHLA"
while flag == 0:
    path = os.getcwd() + "/../ds/SUPERFLOAT/" + name_list[i]
    # print(path)
    if not os.path.exists(path):
        i += 1
        continue

    ds = nc.Dataset(path)
    if var_name not in ds.variables.keys():
        i += 1
        continue
    else:
        flag = 1
        variables = ds.variables
        # print(variables)
        qf = ds[f"{var_name}_QC"][:].data[:]

        if 2 not in qf:
            flag = 0

        var_df = ds[var_name][:].data[:]
        pres_var = ds[f"PRES_{var_name}"][:].data[:]

print(qf)

variable = discretize(pres_var, var_df)
# print("=====")
print(variable)
