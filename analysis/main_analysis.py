from analysis.profile import profile_season_ga
# from analysis.bp import seasonal_and_geographic_bp
from analysis.rmse import *
from analysis.scatter_error import plot_scatter_paper
from analysis.comparison_architecture import reconstruction_profiles


dict_models = {
    "NITRATE": ["2023-12-16", 100],
    "CHLA": ["2023-12-17", 150],
    "BBP700": ["2023-12-15", 125]
}

for var in ["NITRATE", "CHLA", "BBP700"]:
    date = dict_models[var][0]
    epoch = dict_models[var][1]
    # call the function to reconstruct the vertical profiles
    reconstruction_profiles(var, date, epoch, "test")
    # call the function to display the scatter error
    plot_scatter_paper(var, date, epoch, "test")

