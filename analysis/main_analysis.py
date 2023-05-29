# from analysis.profile import *
# from analysis.bp import seasonal_and_geographic_bp
# from analysis.rmse import *
from analysis.scatter_error import plot_scatter_paper
# from analysis.comparison_architecture import *
# from maps import *
# from all_toghether import *

# reconstruction_profile_MLP("NITRATE", "W", "2023-04-04_", 50, "test")

dict_models = {
    "NITRATE": ["2023-04-04_", 50],
    "CHLA": ["2023-03-29", 150],
    "BBP700": ["2023-03-29", 100]
}
# my_var = "NITRATE"
# plot_med(my_var, dict_models[my_var][0], dict_models[my_var][1], "train")

for var in ["NITRATE", "CHLA", "BBP700"]:
    date = dict_models[var][0]
    epoch = dict_models[var][1]

    plot_scatter_paper(var, date, epoch, "test")

for var in ["NITRATE"]:
    date = dict_models[var][0]
    epoch = dict_models[var][1]

    plot_scatter_paper(var, date, epoch, "test")