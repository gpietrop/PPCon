from analysis_utils import *

# i need the rmse

# i need the standard deviation

# i need the number of samples


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


def make_dim_scatter(a_list):
    list_scatter_dim = [250 if loss < 0.3 else 500 if 0.3 < loss < 0.6 else 1000 for loss in a_list]
    return list_scatter_dim


def plot_scatter(variable, date_model, epoch_model, mode):
    list_loss, list_number_samples = seasonal_and_geographic_rmse(variable, date_model, epoch_model, "train")
    list_std = seasonal_and_geographic_std(variable, date_model, epoch_model, "train")

    plt.scatter(list_std[0], list_number_samples[0], s=make_dim_scatter(list_loss[0]),
                c=list(dict_color.values()), marker="^", label="winter")  # winter
    plt.scatter(list_std[1], list_number_samples[1], s=make_dim_scatter(list_loss[1]),
                c=list(dict_color.values()), marker="v", label="spring")  # spring
    plt.scatter(list_std[2], list_number_samples[2], s=make_dim_scatter(list_loss[2]),
                c=list(dict_color.values()), marker="<", label="summer")  # summer
    plt.scatter(list_std[3], list_number_samples[3], s=make_dim_scatter(list_loss[3]),
                c=list(dict_color.values()), marker=">", label="autumn")  # autumn
    plt.legend()
    plt.xlabel("standard deviation")
    plt.ylabel("number of samples (training set)")

    plt.show()

    return
