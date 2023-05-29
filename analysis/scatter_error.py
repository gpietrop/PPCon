from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from utils_analysis import *
import matplotlib

dict_ga = {'NWM': [[40, 45], [-2, 9.5]],
           'SWM': [[32, 40], [-2, 9.5]],
           'TIR': [[37, 45], [9.5, 16]],
           'ION': [[30, 37], [9.5, 22]],
           'LEV': [[30, 37], [22, 36]]}

dict_season = {'W': [0, 91], 'SP': [92, 182], 'SU': [183, 273], 'A': [274, 365]}


def seasonal_ga_variance(season, ga, lat_list, lon_list, day_rad_list, generated_var_list, measured_var_list):
    len_profile = int(generated_var_list[0].size(dim=0))

    generated_profile = []
    # measured_profile = []
    number_samples = len(generated_var_list)
    number_seasonal_samples = 0
    for i in range(number_samples):
        day_sample = from_day_rad_to_day(day_rad=day_rad_list[i])
        if dict_season[season][0] <= day_sample <= dict_season[season][1]:
            if dict_ga[ga][0][0] <= lat_list[i] <= dict_ga[ga][0][1] and dict_ga[ga][1][0] <= lon_list[i] <= \
                    dict_ga[ga][1][1]:
                number_seasonal_samples += 1
                generated_profile.append(generated_var_list[i])
                # measured_profile.append(measured_var_list[i])

    std_reconstruction = torch.zeros(len_profile)
    for k in range(len_profile):
        reconstruction_depth = []
        for prof in generated_profile:
            reconstruction_depth.append(prof[k].item())
        # print(reconstruction_depth)
        std_reconstruction[k] = np.std(reconstruction_depth)

    return torch.mean(std_reconstruction)


def seasonal_and_geographic_std(variable, date_model, epoch_model, mode):
    path_analysis = os.getcwd() + f"/../results/{variable}/{date_model}/"
    if not os.path.exists(path_analysis):
        os.mkdir(path_analysis)

    list_std = [[0 for _ in range(len(list(dict_ga.keys())))] for _ in range(len(list(dict_season.keys())))]

    lat_list, lon_list, day_rad_list, generated_var_list, measured_var_list = get_reconstruction(variable, date_model,
                                                                                                 epoch_model, mode)
    for index_s in range(len(list(dict_season.keys()))):
        season = list(dict_season.keys())[index_s]
        for index_ga in range(len(list(dict_ga.keys()))):
            ga = list(dict_ga.keys())[index_ga]
            std_sample = seasonal_ga_variance(season, ga, lat_list, lon_list, day_rad_list, generated_var_list,
                                              measured_var_list)
            list_std[index_s][index_ga] += std_sample

    # print(list_std)
    return list_std


def seasonal_and_geographic_rmse(variable, date_model, epoch_model, mode):
    path_analysis = os.getcwd() + f"/../results/{variable}/{date_model}/"
    if not os.path.exists(path_analysis):
        os.mkdir(path_analysis)

    list_loss = [[0 for _ in range(len(list(dict_ga.keys())))] for _ in range(len(list(dict_season.keys())))]
    list_number_samples = [[0 for _ in range(len(list(dict_ga.keys())))] for _ in range(len(list(dict_season.keys())))]

    lat_list, lon_list, day_rad_list, generated_var_list, measured_var_list = get_reconstruction(variable, date_model,
                                                                                                 epoch_model, mode)

    number_samples = len(generated_var_list)
    for index_s in range(len(list(dict_season.keys()))):
        season = list(dict_season.keys())[index_s]
        for index_ga in range(len(list(dict_ga.keys()))):
            ga = list(dict_ga.keys())[index_ga]
            for i in range(number_samples):
                day_sample = from_day_rad_to_day(day_rad=day_rad_list[i])
                if dict_season[season][0] <= day_sample <= dict_season[season][1]:
                    if dict_ga[ga][0][0] <= lat_list[i] <= dict_ga[ga][0][1] and dict_ga[ga][1][0] <= lon_list[i] <= \
                            dict_ga[ga][1][1]:
                        loss_sample = mse_loss(generated_var_list[i], measured_var_list[i])
                        list_loss[index_s][index_ga] += loss_sample
                        list_number_samples[index_s][index_ga] += 1

    list_loss = np.array(list_loss) / np.array(list_number_samples)

    return list_loss, list_number_samples


def make_dim_scatter(a_list, variable):
    if variable == "NITRATE":
        list_scatter_dim = [75 if loss < 0.3 else 250 if 0.3 < loss < 0.6 else 650 for loss in a_list]
    if variable == "CHLA":
        list_scatter_dim = [25 if loss < 0.01 else 150 if 0.01 < loss < 0.05 else 500 for loss in a_list]
    if variable == "BBP700":
        list_scatter_dim = [25 if loss < 0.00000005 else 150 if 0.00000005 < loss < 0.00000015 else 500 for loss in
                            a_list]

    return list_scatter_dim


def plot_scatter(variable, date_model, epoch_model, mode):
    path_analysis = os.getcwd() + f"/../results/{variable}/{date_model}/fig/"
    if not os.path.exists(path_analysis):
        os.mkdir(path_analysis)

    # fig, ax = plt.subplots(figsize=(8, 5))
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])

    pal = sns.color_palette("muted")
    dict_color = {'NWM': pal[0], 'SWM': pal[1], 'TYR': pal[2], 'ION': pal[3], 'LEV': pal[4]}

    list_loss, list_number_samples = seasonal_and_geographic_rmse(variable, date_model, epoch_model, "train")
    list_std = seasonal_and_geographic_std(variable, date_model, epoch_model, "train")

    ax.scatter(list_std[0], list_number_samples[0], s=make_dim_scatter(list_loss[0], variable),
               c=list(dict_color.values()), marker="^", label="winter", alpha=0.6)  # winter
    ax.scatter(list_std[1], list_number_samples[1], s=make_dim_scatter(list_loss[1], variable),
               c=list(dict_color.values()), marker="s", label="spring", alpha=0.6)  # spring
    ax.scatter(list_std[2], list_number_samples[2], s=make_dim_scatter(list_loss[2], variable),
               c=list(dict_color.values()), marker="o", label="summer", alpha=0.6)  # summer
    ax.scatter(list_std[3], list_number_samples[3], s=make_dim_scatter(list_loss[3], variable),
               c=list(dict_color.values()), marker="D", label="autumn", alpha=0.6)  # autumn

    # legend 1 -- MAE dimension
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8),
                       Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=13),
                       Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=18)
                       ]

    if variable == "NITRATE":
        un_meas = dict_unit_measure["NITRATE"]
        lg1 = ax.legend(legend_elements, ["MAE<0.3", "0.3<MAE<0.6", "MAE>0.6"],
                        bbox_to_anchor=(1.0, 1.0), loc="upper left", title=f"MAE [{un_meas}]")
    if variable == "CHLA":
        un_meas = dict_unit_measure["CHLA"]
        lg1 = ax.legend(legend_elements, ["MAE<0.01", "0.01<MAE<0.05", "MAE>0.05"],
                        bbox_to_anchor=(1.0, 1.0), loc="upper left", title=f"MAE [{un_meas}]")
    if variable == "BBP700":
        un_meas = dict_unit_measure["BBP700"]
        lg1 = ax.legend(legend_elements, ["MAE<0.5e-8", "0.5e-8<MAE<1.5e-8", "MAE>1.5e-8"],
                        bbox_to_anchor=(1.0, 1.0), loc="upper left", title=f"MAE [{un_meas}]")

    # legend 2 -- season
    handles, labels = ax.get_legend_handles_labels()
    lg2 = ax.legend(handles, labels, bbox_to_anchor=(1.0, 0.7), loc="upper left", title="season", markerscale=0.5)

    for marker in lg2.legendHandles:
        marker.set_color("k")

    # legend 3 -- ga
    legend_elements = [Patch(facecolor=list(dict_color.values())[0], edgecolor='k', label=list(dict_color.keys())[0]),
                       Patch(facecolor=list(dict_color.values())[1], edgecolor='k', label=list(dict_color.keys())[1]),
                       Patch(facecolor=list(dict_color.values())[2], edgecolor='k', label=list(dict_color.keys())[2]),
                       Patch(facecolor=list(dict_color.values())[3], edgecolor='k', label=list(dict_color.keys())[3]),
                       Patch(facecolor=list(dict_color.values())[4], edgecolor='k', label=list(dict_color.keys())[4]),
                       ]
    lg3 = ax.legend(handles=legend_elements, bbox_to_anchor=(1.0, 0.35), loc="upper left", title="geographic area")

    ax.add_artist(lg1)
    ax.add_artist(lg2)
    ax.add_artist(lg3)

    # ax.xaxis.set_major_formatter(FormatStrFormatter('% 1.2f'))
    import matplotlib
    ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0e}'))
    plt.xlabel(f"standard deviation [{un_meas}]")
    plt.ylabel("number of samples (training set)")

    plt.savefig(f"{path_analysis}scatter_{mode}_{epoch_model}.png")
    # plt.show()
    plt.close()

    return


def plot_scatter_paper(variable, date_model, epoch_model, mode):

    path_analysis = os.getcwd() + f"/../results/{variable}/{date_model}/fig/"
    if not os.path.exists(path_analysis):
        os.mkdir(path_analysis)

    pal = sns.color_palette("muted")
    dict_color = {'NWM': pal[0], 'SWM': pal[1], 'TYR': pal[2], 'ION': pal[3], 'LEV': pal[4]}

    fig, ax = plt.subplots()
    # fig = plt.figure(figsize=(8, 5))

    list_loss, list_number_samples = seasonal_and_geographic_rmse(variable, date_model, epoch_model, "train")
    list_std = seasonal_and_geographic_std(variable, date_model, epoch_model, "train")

    ax.scatter(list_std[0], list_number_samples[0], s=make_dim_scatter(list_loss[0], variable),
               c=list(dict_color.values()), marker="^", label="winter", alpha=0.6)  # winter
    ax.scatter(list_std[1], list_number_samples[1], s=make_dim_scatter(list_loss[1], variable),
               c=list(dict_color.values()), marker="s", label="spring", alpha=0.6)  # spring
    ax.scatter(list_std[2], list_number_samples[2], s=make_dim_scatter(list_loss[2], variable),
               c=list(dict_color.values()), marker="o", label="summer", alpha=0.6)  # summer
    ax.scatter(list_std[3], list_number_samples[3], s=make_dim_scatter(list_loss[3], variable),
               c=list(dict_color.values()), marker="D", label="autumn", alpha=0.6)  # autumn

    # legend 1 -- MAE dimension
    # handles, labels = scatter1.legend_elements(prop="sizes", alpha=0.6)
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=6),
                       Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=11),
                       Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=16)
                       ]

    if variable == "NITRATE":
        un_meas = dict_unit_measure["NITRATE"]
        lg1 = ax.legend(legend_elements, ["MSE<" + r"$0.3$", r"$0.3<$" + "MSE" + r"$<0.6$", "MSE" + r"$>0.6$"],
                        fontsize="10", title=f"MSE [{un_meas}]")
    if variable == "CHLA":
        un_meas = dict_unit_measure["CHLA"]
        lg1 = ax.legend(legend_elements, ["MSE<" + r"$0.01$", r"$0.01<$" + "MSE" + r"$<0.05$", "MSE" + r"$>0.05$"],
                        fontsize="10", title=f"MSE [{un_meas}]")
    if variable == "BBP700":
        un_meas = dict_unit_measure["BBP700"]
        lg1 = ax.legend(legend_elements, ["MSE<" + r"$0.5e^{-8}$", r"$0.5e^{-8}<$" + "MSE" + r"$<1.5e^{-8}$",
                                          "MSE" + r"$>1.5e^{-8}$"],
                        fontsize="10", title=f"MSE [{un_meas}]")

    # ax.xaxis.set_major_formatter(FormatStrFormatter('% 1.2f'))
    if variable == "BBP700":
        ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0e}'))
    plt.xlabel(f"standard deviation [{un_meas}]")
    plt.ylabel("number of samples (training set)")

    if variable == "CHLA":
        plt.title("CHLOROPHYLL")
    else:
        plt.title(variable)

    plt.tight_layout()

    plt.savefig(f"{path_analysis}scatter_{mode}_{epoch_model}.png")
    plt.savefig(os.getcwd() + f"/../results/paper_fig/scatter_{variable}_{mode}.png")

    # plt.show()
    plt.close()

    return
