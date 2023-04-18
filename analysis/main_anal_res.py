from analysis.analysis_ds_reconstructed import *

# get_reconstruction_comparison("W", "2023-03-29", 50, "test")
dict_models = {
    "NITRATE": ["2023-04-04_", 50],
    "CHLA": ["2023-03-29", 150],
    "BBP700": ["2023-03-29", 100]
}
for var in ["NITRATE"]:
    date = dict_models[var][0]
    epoch = dict_models[var][1]

    # seasonal_and_geographic_bp(var, date, epoch, "train")
    profile_error(var, date, epoch, "test")
    # profile_variance(var, date, epoch, "train")
    # profile_variance(var, date, epoch, "test")
    # profile(var, date, epoch, "test")
# seasonal_and_geographic_rmse(var, date, epoch_model=100, mode="train")
# seasonal_and_geographic_rmse(var, date, epoch_model=100, mode="test")
"""
for epoch in [100]:

    # plot_seasonal_ga_med("SU", var, date, epoch, "test")
    seasonal_and_geographic_bp(var, date, epoch, "train")
    seasonal_and_geographic_bp(var, date, epoch, "test")
    profile(var, date, epoch, "train")
    profile(var, date, epoch, "test")
"""