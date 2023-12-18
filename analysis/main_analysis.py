from analysis.profile import profile_season_ga
# from analysis.bp import seasonal_and_geographic_bp
from analysis.rmse import *
from analysis.scatter_error import plot_scatter_paper, plot_scatter_paper_log
from analysis.comparison_architecture import *
# from maps import *
# from all_toghether import *

# reconstruction_profile_MLP("NITRATE", "all", "2023-04-04_", 50, "test")

dict_models = {
    "NITRATE": ["2023-12-16_", 100],
    "CHLA": ["2023-12-17", 100],
    "BBP700": ["2023-12-17", 175]
}
# my_var = "NITRATE"
# plot_med(my_var, dict_models[my_var][0], dict_models[my_var][1], "train")

for var in ["NITRATE"]:
    date = dict_models[var][0]
    epoch = dict_models[var][1]
    reconstruction_profile_MLP(var, date, epoch, "test")
    # reconstruction_profiles(var, date, epoch, "test")
    # reconstruction_profile(var, date, epoch, "test")
    # seasonal_and_geographic_rmse(var, date, epoch, "test")
    # seasonal_and_geographic_rmse(var, date, epoch, "test")
    # plot_scatter_paper(var, date, epoch, "test")
    # plot_scatter_paper_log(var, date, epoch, "test")

for var in ["NITRATE"]:
    date = dict_models[var][0]
    epoch = dict_models[var][1]

