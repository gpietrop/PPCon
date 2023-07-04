import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from dadapy.data import Data
import matplotlib.pyplot as plt
from dadapy.plot import plot_SLAn, plot_MDS, plot_matrix, get_dendrogram, plot_DecGraph

from analysis.utils_analysis import from_day_rad_to_day
from utils_clustering import make_ds
from dataset_clustering import FloatDataset
from plot_clustering import plot_density_points, plot_adp_clustering, plot_clustering_coordinates

# make_ds("SUPERFLOAT")
ds_enhanced_path = os.getcwd() + "/../ds/clustering/ds_sf_clustering_enhanced.csv"

dataset = FloatDataset(ds_enhanced_path)
ds = DataLoader(dataset, shuffle=True)

test_frac = 0.5
test_size = int(test_frac * len(dataset))
toy_dataset, _ = torch.utils.data.random_split(dataset, [test_size, len(dataset) - test_size])
toy_loader = DataLoader(toy_dataset, shuffle=True)

# generate the dataset for the clustering procedure
ds_clustering = list()

lat_values = []
lon_values = []
for year, day_rad, lat, lon, temp, psal, doxy, nitrate, chla, BBP700, name_float in toy_loader:
    season = "W"
    dict_season = {'W': [0, 91], 'SP': [92, 182], 'SU': [183, 273], 'A': [274, 365]}
    day_sample = from_day_rad_to_day(day_rad=day_rad)

    if dict_season[season][0] <= day_sample <= dict_season[season][1]:
        lat_values.append(lat)
        lon_values.append(lon)

        nitrate = torch.squeeze(nitrate[:100:4]).numpy()
        doxy = torch.squeeze(doxy[::4]).numpy()

        ds_clustering.append(np.concatenate((doxy, nitrate)))

ds_clustering = np.array(ds_clustering)
# print(ds_clustering)
print(len(ds_clustering))

# initialise the "Data" class with a set of coordinates
data = Data(ds_clustering)

# compute distances up to the 100th nearest neighbour
data.compute_distances(maxk=100)

# compute the intrinsic dimension using the 2NN estimator
intrinsic_dim, _, intrinsic_dim_err = data.compute_id_2NN()

# check the value of the intrinsic dimension found
print(data.intrinsic_dim)

# compute the density of all points using a simple kNN estimator
log_den, log_den_err = data.compute_density_kNN(k=15)
# plot_density_points(X=ds_clustering, log_den=log_den)

# Compute the so-called decison graph and plot it (Note that the density has been computed in previous steps).
data.compute_DecGraph()
plot_DecGraph(data)

# find the statistically significant peaks of the density profile computed previously
data.compute_clustering_ADP(Z=1.5)
print(data.N_clusters)
plot_adp_clustering(data=data)
# print(data.cluster_assignment)

plot_clustering_coordinates(lat_values=lat_values, lon_values=lon_values, clusters=data.cluster_assignment)




"""
    print(nitrate)

    plt.plot(BBP700[0, :].detach().numpy())
    # plt.plot(output[0, 0, :].detach().numpy(), label="measured")
    plt.legend()
    # plt.savefig(dir_profile + f"/profile_{year}_{day_rad}_{round(lat, 2)}_{round(lon, 2)}.png")
    plt.show()
    plt.close()
    """
