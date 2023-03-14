import os

import netCDF4 as nc
import pandas as pd
import matplotlib.pyplot as plt

from make_ds.make_superfloat_ds import discretize
from discretization import dict_max_pressure, dict_interval

path = os.getcwd() + '/../ds/SUPERFLOAT/Float_Index.txt'
name_list = pd.read_csv(path, header=None).to_numpy()[:, 0].tolist()

i = 290
flag = 0
var_name = "BBP700"
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
        qf = ds[f"{var_name}"][:].data[:]

        # if 2 not in qf:
        #    flag = 0

        var_df = ds[var_name][:].data[:]
        pres_var = ds[f"PRES_{var_name}"][:].data[:]

print(qf)
print(pres_var)
variable = discretize(pres_var, var_df, max_pres=dict_max_pressure[var_name], interval=dict_interval[var_name])
# print("=====")
print(variable)
plt.plot(variable)
plt.show()
plt.close()
