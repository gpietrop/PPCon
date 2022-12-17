import os

import netCDF4 as nc
import pandas as pd

from make_ds.make_ds import discretize

path = os.getcwd() + '/SUPERFLOAT/Float_Index.txt'
name_list = pd.read_csv(path, header=None).to_numpy()[:, 0].tolist()

i = 0
flag = 0
while flag == 0:
    path = "SUPERFLOAT/" + name_list[i]
    print(path)
    if not os.path.exists(path):
        i += 1
        continue

    ds = nc.Dataset(path)
    if "NITRATE" not in ds.variables.keys():
        i += 1
        continue
    else:
        flag = 1
        nitrate_df = ds["NITRATE"][:].data[:]
        pres_nitrate = ds["PRES_NITRATE"][:].data[:]

nitrate = discretize(pres_nitrate, nitrate_df)
print(nitrate)