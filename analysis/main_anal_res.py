# from analysis.profile import *
# from analysis.rmse import *
# from analysis.comparison_architecture import *
# from maps import *
from all_toghether import *
# get_reconstruction_comparison_presentation("NITRATE", "W", "2023-04-04_", 50, "test")

dict_models = {
    "NITRATE": ["2023-04-04_", 50],
    "CHLA": ["2023-03-29", 150],
    "BBP700": ["2023-03-29", 100]
}
my_var = "NITRATE"
# plot_med(my_var, dict_models[my_var][0], dict_models[my_var][1], "train")

for var in ["CHLA"]:
    date = dict_models[var][0]
    epoch = dict_models[var][1]
    # for ga in ["NWM", "SWM", "TYR", "ION", "LEV"]:
    #     ga_profile(ga, var, date, epoch, "test")
    # ga_variance("NWM", var, date, epoch, "test")

    plot_scatter(var, date, epoch, "test")

    # all_profile(var, date, epoch, "test")
    # profile_efficency(var, date, epoch, "test")
    # profile_error(var, date, epoch, "test")
    # seasonal_and_geographic_rmse(var, date, epoch, "train")
    # seasonal_and_geographic_bp(var, date, epoch, "train")
    # profile(var, date, epoch, "test")
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