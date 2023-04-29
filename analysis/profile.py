from analysis_utils import *


def all_profile(variable, date_model, epoch_model, mode):
    path_analysis = os.getcwd() + f"/../results/{variable}/{date_model}/fig/profile_{mode}/"
    if not os.path.exists(path_analysis):
        os.mkdir(path_analysis)

    lat_list, lon_list, day_rad_list, generated_var_list, measured_var_list = get_reconstruction(variable, date_model,
                                                                                                 epoch_model, mode)
    number_samples = len(generated_var_list)

    for index_sample in range(number_samples):
        lat = lat_list[index_sample]
        lon = lon_list[index_sample]
        generated_profile = generated_var_list[index_sample]
        measured_profile = measured_var_list[index_sample]

        max_pres = dict_max_pressure[variable]
        depth = np.linspace(0, max_pres, len(generated_profile.detach().numpy()))
        plt.plot(measured_profile.detach().numpy(), depth, lw=2.5,
                 color="green", label=f"measured {variable}")
        plt.plot(generated_profile.detach().numpy(), depth, lw=2.5,
                 color="blue", linestyle="dashed", label=f"convolutional {variable}")
        plt.gca().invert_yaxis()

        plt.xlabel(f"{variable} [{dict_unit_measure[variable]}]")
        plt.ylabel('depth [m]')

        plt.legend()
        plt.savefig(f"{path_analysis}profile_{round(lat, 2)}_{round(lon, 2)}.png")
        plt.close()

    return


def seasonal_profile(season, variable, date_model, epoch_model, mode):
    dict_season = {'W': [0, 91], 'SP': [92, 182], 'SU': [183, 273], 'A': [274, 365]}
    lat_list, lon_list, day_rad_list, generated_var_list, measured_var_list = get_reconstruction(variable, date_model,
                                                                                                 epoch_model, mode)
    len_profile = int(generated_var_list[0].size(dim=0))

    generated_profile = torch.zeros(len_profile)
    measured_profile = torch.zeros(len_profile)
    number_samples = len(generated_var_list)
    number_seasonal_samples = 0
    for index_sample in range(number_samples):
        day_sample = from_day_rad_to_day(day_rad=day_rad_list[index_sample])
        if dict_season[season][0] <= day_sample <= dict_season[season][1]:
            number_seasonal_samples += 1
            generated_profile = generated_profile + generated_var_list[index_sample]
            measured_profile = measured_profile + measured_var_list[index_sample]

    generated_profile = generated_profile / number_seasonal_samples
    measured_profile = measured_profile / number_seasonal_samples

    max_pres = dict_max_pressure[variable]
    depth = np.linspace(0, max_pres, len(generated_profile.detach().numpy()))
    plt.plot(generated_profile.detach().numpy(), depth, label=f"generated {variable}")
    plt.plot(measured_profile.detach().numpy(), depth, label=f"measured {variable}")
    plt.gca().invert_yaxis()

    plt.legend()
    plt.show()
    plt.close()

    return


def ga_profile(ga, variable, date_model, epoch_model, mode):
    path_analysis = os.getcwd() + f"/../results/{variable}/{date_model}/fig/"
    if not os.path.exists(path_analysis):
        os.mkdir(path_analysis)
    path_analysis = path_analysis + "ga_prof/"
    if not os.path.exists(path_analysis):
        os.mkdir(path_analysis)

    dict_ga = {'NWM': [[40, 45], [-2, 9.5]],
               'SWM': [[32, 40], [-2, 9.5]],
               'TIR': [[37, 45], [9.5, 16]],
               'ION': [[30, 37], [9.5, 22]],
               'LEV': [[30, 37], [22, 36]]}

    lat_list, lon_list, day_rad_list, generated_var_list, measured_var_list = get_reconstruction(variable, date_model,
                                                                                                 epoch_model, mode)
    len_profile = int(generated_var_list[0].size(dim=0))

    generated_profile = torch.zeros(len_profile)
    measured_profile = torch.zeros(len_profile)
    number_samples = len(generated_var_list)
    number_ga_samples = 0
    for index_sample in range(number_samples):
        if dict_ga[ga][0][0] <= lat_list[index_sample] <= dict_ga[ga][0][1] and dict_ga[ga][1][0] <= lon_list[
            index_sample] <= dict_ga[ga][1][1]:
            number_ga_samples += 1

            generated_profile = generated_profile + generated_var_list[index_sample]
            measured_profile = measured_profile + measured_var_list[index_sample]

    generated_profile = generated_profile / number_ga_samples
    measured_profile = measured_profile / number_ga_samples

    max_pres = dict_max_pressure[variable]
    depth = np.linspace(0, max_pres, len(generated_profile.detach().numpy()))
    plt.plot(generated_profile.detach().numpy(), depth,
             linestyle="dashed", color=dict_color[ga], label=f"CNN generated")
    plt.plot(measured_profile.detach().numpy(), depth,
             linestyle="solid", color=dict_color[ga], label=f"measured")
    plt.gca().invert_yaxis()

    plt.title(ga)
    plt.xlabel(f"{variable} [{dict_unit_measure[variable]}]")
    plt.ylabel('depth [m]')

    plt.legend()
    plt.savefig(f"{path_analysis}profile_mean_{ga}_{mode}_{epoch_model}.png")
    plt.close()

    return


def seasonal_ga_profile(season, ga, lat_list, lon_list, day_rad_list, generated_var_list, measured_var_list):
    dict_ga = {'NWM': [[40, 45], [-2, 9.5]],
               'SWM': [[32, 40], [-2, 9.5]],
               'TIR': [[37, 45], [9.5, 16]],
               'ION': [[30, 37], [9.5, 22]],
               'LEV': [[30, 37], [22, 36]]}

    dict_season = {'W': [0, 91], 'SP': [92, 182], 'SU': [183, 273], 'A': [274, 365]}

    len_profile = int(generated_var_list[0].size(dim=0))

    generated_profile = torch.zeros(len_profile)
    measured_profile = torch.zeros(len_profile)
    number_samples = len(generated_var_list)
    number_seasonal_samples = 0
    for i in range(number_samples):
        day_sample = from_day_rad_to_day(day_rad=day_rad_list[i])
        if dict_season[season][0] <= day_sample <= dict_season[season][1]:
            if dict_ga[ga][0][0] <= lat_list[i] <= dict_ga[ga][0][1] and dict_ga[ga][1][0] <= lon_list[i] <= \
                    dict_ga[ga][1][1]:
                number_seasonal_samples += 1
                generated_profile = generated_profile + generated_var_list[i]
                measured_profile = measured_profile + measured_var_list[i]

    generated_profile = generated_profile / number_seasonal_samples
    measured_profile = measured_profile / number_seasonal_samples

    return generated_profile, measured_profile


def profile(variable, date_model, epoch_model, mode):
    path_analysis = os.getcwd() + f"/../results/{variable}/{date_model}/fig/"
    if not os.path.exists(path_analysis):
        os.mkdir(path_analysis)

    dict_ga = {'NWM': [[40, 45], [-2, 9.5]],
               'SWM': [[32, 40], [-2, 9.5]],
               'TIR': [[37, 45], [9.5, 16]],
               'ION': [[30, 37], [9.5, 22]],
               'LEV': [[30, 37], [22, 36]]}

    dict_season = {'W': [0, 91], 'SP': [92, 182], 'SU': [183, 273], 'A': [274, 365]}

    lat_list, lon_list, day_rad_list, generated_var_list, measured_var_list = get_reconstruction(variable, date_model,
                                                                                                 epoch_model, mode)

    fig, axs = plt.subplots(2, 2)

    for ga in list(dict_ga.keys()):
        generated, measured = seasonal_ga_profile("W", ga, lat_list, lon_list, day_rad_list, generated_var_list,
                                                  measured_var_list)
        max_pres = dict_max_pressure[variable]
        depth = np.linspace(0, max_pres, len(generated.detach().numpy()))

        axs[0, 0].plot(generated.detach().numpy(), depth,
                       linestyle="dashed", color=dict_color[ga])
        axs[0, 0].plot(measured.detach().numpy(), depth,
                       label=ga, linestyle="solid", color=dict_color[ga])
        axs[0, 0].invert_yaxis()

        generated, measured = seasonal_ga_profile("SP", ga, lat_list, lon_list, day_rad_list, generated_var_list,
                                                  measured_var_list)
        axs[0, 1].plot(generated.detach().numpy(), depth,
                       linestyle="dashed", color=dict_color[ga])
        axs[0, 1].plot(measured.detach().numpy(), depth,
                       label=ga, linestyle="solid", color=dict_color[ga])
        axs[0, 1].invert_yaxis()

        generated, measured = seasonal_ga_profile("SU", ga, lat_list, lon_list, day_rad_list, generated_var_list,
                                                  measured_var_list)
        axs[1, 0].plot(generated.detach().numpy(), depth,
                       linestyle="dashed", color=dict_color[ga])
        axs[1, 0].plot(measured.detach().numpy(), depth,
                       label=ga, linestyle="solid", color=dict_color[ga])
        axs[1, 0].invert_yaxis()

        generated, measured = seasonal_ga_profile("A", ga, lat_list, lon_list, day_rad_list, generated_var_list,
                                                  measured_var_list)
        axs[1, 1].plot(generated.detach().numpy(), depth,
                       linestyle="dashed", color=dict_color[ga])
        axs[1, 1].plot(measured.detach().numpy(), depth,
                       label=ga, linestyle="solid", color=dict_color[ga])
        axs[1, 1].invert_yaxis()

    axs[0, 0].set_title(list(dict_season.keys())[0])
    axs[0, 1].set_title(list(dict_season.keys())[1])
    axs[1, 0].set_title(list(dict_season.keys())[2])
    axs[1, 1].set_title(list(dict_season.keys())[3])

    for ax in axs.flat:
        ax.set(xlabel=f"{variable} ({dict_unit_measure[variable]})", ylabel='depth')
        # ax.set_xticks(range(1, len(list(dict_ga.keys())) + 1), list(dict_ga.keys()))

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize="5")

    plt.savefig(f"{path_analysis}profile_comparison_{mode}_{epoch_model}.png")

    # plt.show()
    plt.close()

    return


def profile_error(variable, date_model, epoch_model, mode):
    path_analysis = os.getcwd() + f"/../results/{variable}/{date_model}/fig/"
    if not os.path.exists(path_analysis):
        os.mkdir(path_analysis)

    dict_ga = {'NWM': [[40, 45], [-2, 9.5]],
               'SWM': [[32, 40], [-2, 9.5]],
               'TIR': [[37, 45], [9.5, 16]],
               'ION': [[30, 37], [9.5, 22]],
               'LEV': [[30, 37], [22, 36]]}

    dict_season = {'W': [0, 91], 'SP': [92, 182], 'SU': [183, 273], 'A': [274, 365]}

    lat_list, lon_list, day_rad_list, generated_var_list, measured_var_list = get_reconstruction(variable, date_model,
                                                                                                 epoch_model, mode)

    fig, axs = plt.subplots(2, 2)

    for ga in list(dict_ga.keys()):
        generated, measured = seasonal_ga_profile("W", ga, lat_list, lon_list, day_rad_list, generated_var_list,
                                                  measured_var_list)
        max_pres = dict_max_pressure[variable]
        depth = np.linspace(0, max_pres, len(generated.detach().numpy()))

        axs[0, 0].plot(np.abs(measured.detach().numpy()) -
                       np.abs(generated.detach().numpy()), depth,
                       label=ga, linestyle="solid", color=dict_color[ga])
        axs[0, 0].invert_yaxis()

        generated, measured = seasonal_ga_profile("SP", ga, lat_list, lon_list, day_rad_list, generated_var_list,
                                                  measured_var_list)
        axs[0, 1].plot(np.abs(measured.detach().numpy()) -
                       np.abs(generated.detach().numpy()), depth,
                       label=ga, linestyle="solid", color=dict_color[ga])
        axs[0, 1].invert_yaxis()

        generated, measured = seasonal_ga_profile("SU", ga, lat_list, lon_list, day_rad_list, generated_var_list,
                                                  measured_var_list)
        axs[1, 0].plot(np.abs(measured.detach().numpy()) -
                       np.abs(generated.detach().numpy()), depth,
                       label=ga, linestyle="solid", color=dict_color[ga])
        axs[1, 0].invert_yaxis()

        generated, measured = seasonal_ga_profile("A", ga, lat_list, lon_list, day_rad_list, generated_var_list,
                                                  measured_var_list)
        axs[1, 1].plot(np.abs(measured.detach().numpy()) -
                       np.abs(generated.detach().numpy()), depth,
                       label=ga, linestyle="solid", color=dict_color[ga])
        axs[1, 1].invert_yaxis()

    axs[0, 0].set_title(list(dict_season.keys())[0])
    axs[0, 1].set_title(list(dict_season.keys())[1])
    axs[1, 0].set_title(list(dict_season.keys())[2])
    axs[1, 1].set_title(list(dict_season.keys())[3])

    for ax in axs.flat:
        ax.set(xlabel=f"{variable} ({dict_unit_measure[variable]})", ylabel='depth')
        # ax.set_xticks(range(1, len(list(dict_ga.keys())) + 1), list(dict_ga.keys()))

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    # for ax in axs.flat:
    #    ax.label_outer()

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize="5")

    plt.savefig(f"{path_analysis}profile_error_{mode}_{epoch_model}.png")

    # plt.show()
    plt.close()

    return


def seasonal_ga_list(season, ga, lat_list, lon_list, day_rad_list, generated_var_list, measured_var_list):
    dict_ga = {'NWM': [[40, 45], [-2, 9.5]],
               'SWM': [[32, 40], [-2, 9.5]],
               'TIR': [[37, 45], [9.5, 16]],
               'ION': [[30, 37], [9.5, 22]],
               'LEV': [[30, 37], [22, 36]]}

    dict_season = {'W': [0, 91], 'SP': [92, 182], 'SU': [183, 273], 'A': [274, 365]}

    generated_profile = []
    measured_profile = []
    number_samples = len(generated_var_list)
    for i in range(number_samples):
        day_sample = from_day_rad_to_day(day_rad=day_rad_list[i])
        if dict_season[season][0] <= day_sample <= dict_season[season][1]:
            if dict_ga[ga][0][0] <= lat_list[i] <= dict_ga[ga][0][1] and dict_ga[ga][1][0] <= lon_list[i] <= \
                    dict_ga[ga][1][1]:
                generated_profile.append(generated_var_list[i])
                measured_profile.append(measured_var_list[i])

    return generated_profile, measured_profile


def seasonal_ga_variance(season, ga, lat_list, lon_list, day_rad_list, generated_var_list, measured_var_list):
    dict_ga = {'NWM': [[40, 45], [-2, 9.5]],
               'SWM': [[32, 40], [-2, 9.5]],
               'TIR': [[37, 45], [9.5, 16]],
               'ION': [[30, 37], [9.5, 22]],
               'LEV': [[30, 37], [22, 36]]}

    dict_season = {'W': [0, 91], 'SP': [92, 182], 'SU': [183, 273], 'A': [274, 365]}

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

    return std_reconstruction



def seasonal_ga_variance(season, ga, lat_list, lon_list, day_rad_list, generated_var_list, measured_var_list):
    dict_ga = {'NWM': [[40, 45], [-2, 9.5]],
               'SWM': [[32, 40], [-2, 9.5]],
               'TIR': [[37, 45], [9.5, 16]],
               'ION': [[30, 37], [9.5, 22]],
               'LEV': [[30, 37], [22, 36]]}

    dict_season = {'W': [0, 91], 'SP': [92, 182], 'SU': [183, 273], 'A': [274, 365]}

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

    return std_reconstruction


def profile_variance(variable, date_model, epoch_model, mode):
    path_analysis = os.getcwd() + f"/../results/{variable}/{date_model}/fig/"
    if not os.path.exists(path_analysis):
        os.mkdir(path_analysis)

    dict_ga = {'NWM': [[40, 45], [-2, 9.5]],
               'SWM': [[32, 40], [-2, 9.5]],
               'TIR': [[37, 45], [9.5, 16]],
               'ION': [[30, 37], [9.5, 22]],
               'LEV': [[30, 37], [22, 36]]}

    dict_season = {'W': [0, 91], 'SP': [92, 182], 'SU': [183, 273], 'A': [274, 365]}

    lat_list, lon_list, day_rad_list, generated_var_list, measured_var_list = get_reconstruction(variable, date_model,
                                                                                                 epoch_model, mode)

    fig, axs = plt.subplots(2, 2)

    for ga in list(dict_ga.keys()):
        variance = seasonal_ga_variance("W", ga, lat_list, lon_list, day_rad_list, generated_var_list,
                                        measured_var_list)
        max_pres = dict_max_pressure[variable]
        depth = np.linspace(0, max_pres, len(variance.detach().numpy()))

        axs[0, 0].plot(variance.detach().numpy(), depth, color=dict_color[ga], label=ga)
        axs[0, 0].invert_yaxis()

        variance = seasonal_ga_variance("SP", ga, lat_list, lon_list, day_rad_list, generated_var_list,
                                        measured_var_list)
        axs[0, 1].plot(variance.detach().numpy(), depth, color=dict_color[ga], label=ga)
        axs[0, 1].invert_yaxis()

        variance = seasonal_ga_variance("SU", ga, lat_list, lon_list, day_rad_list, generated_var_list,
                                        measured_var_list)
        axs[1, 0].plot(variance.detach().numpy(), depth, color=dict_color[ga], label=ga)
        axs[1, 0].invert_yaxis()

        variance = seasonal_ga_variance("A", ga, lat_list, lon_list, day_rad_list, generated_var_list,
                                        measured_var_list)
        axs[1, 1].plot(variance.detach().numpy(), depth, color=dict_color[ga], label=ga)
        axs[1, 1].invert_yaxis()

    axs[0, 0].set_title(list(dict_season.keys())[0])
    axs[0, 1].set_title(list(dict_season.keys())[1])
    axs[1, 0].set_title(list(dict_season.keys())[2])
    axs[1, 1].set_title(list(dict_season.keys())[3])

    for ax in axs.flat:
        ax.set(xlabel=f"{variable} ({dict_unit_measure[variable]})", ylabel='depth')
        # ax.set_xticks(range(1, len(list(dict_ga.keys())) + 1), list(dict_ga.keys()))

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize="5")

    plt.savefig(f"{path_analysis}variance_comparison_{mode}_{epoch_model}.png")

    # plt.show()
    plt.close()

    return


def profile_efficency(variable, date_model, epoch_model, mode):
    path_analysis = os.getcwd() + f"/../results/{variable}/{date_model}/fig/"
    if not os.path.exists(path_analysis):
        os.mkdir(path_analysis)

    dict_ga = {'NWM': [[40, 45], [-2, 9.5]],
               'SWM': [[32, 40], [-2, 9.5]],
               'TIR': [[37, 45], [9.5, 16]],
               'ION': [[30, 37], [9.5, 22]],
               'LEV': [[30, 37], [22, 36]]}

    dict_season = {'W': [0, 91], 'SP': [92, 182], 'SU': [183, 273], 'A': [274, 365]}

    lat_list, lon_list, day_rad_list, generated_var_list, measured_var_list = get_reconstruction(variable, date_model,
                                                                                                 epoch_model, mode)
    fig, axs = plt.subplots(2, 2)

    for ga in list(dict_ga.keys()):
        variance = seasonal_ga_variance("W", ga, lat_list, lon_list, day_rad_list, generated_var_list,
                                        measured_var_list)
        generated_var_list_s_ga, measured_var_list_s_ga = seasonal_ga_list("W", ga, lat_list, lon_list, day_rad_list,
                                                                           generated_var_list, measured_var_list)
        efficency = compute_efficency(generated_var_list_s_ga, measured_var_list_s_ga, variance)

        max_pres = dict_max_pressure[variable]
        depth = np.linspace(0, max_pres, len(variance.detach().numpy()))

        axs[0, 0].plot(efficency.detach().numpy(), depth, color=dict_color[ga], label=ga)
        axs[0, 0].invert_yaxis()

        variance = seasonal_ga_variance("SP", ga, lat_list, lon_list, day_rad_list, generated_var_list,
                                        measured_var_list)
        generated_var_list_s_ga, measured_var_list_s_ga = seasonal_ga_list("SP", ga, lat_list, lon_list, day_rad_list,
                                                                           generated_var_list, measured_var_list)
        efficency = compute_efficency(generated_var_list_s_ga, measured_var_list_s_ga, variance)

        axs[0, 1].plot(efficency.detach().numpy(), depth, color=dict_color[ga], label=ga)
        axs[0, 1].invert_yaxis()

        variance = seasonal_ga_variance("SU", ga, lat_list, lon_list, day_rad_list, generated_var_list,
                                        measured_var_list)
        generated_var_list_s_ga, measured_var_list_s_ga = seasonal_ga_list("SU", ga, lat_list, lon_list, day_rad_list,
                                                                           generated_var_list, measured_var_list)
        efficency = compute_efficency(generated_var_list_s_ga, measured_var_list_s_ga, variance)

        axs[1, 0].plot(efficency.detach().numpy(), depth, color=dict_color[ga], label=ga)
        axs[1, 0].invert_yaxis()

        variance = seasonal_ga_variance("A", ga, lat_list, lon_list, day_rad_list, generated_var_list,
                                        measured_var_list)
        generated_var_list_s_ga, measured_var_list_s_ga = seasonal_ga_list("A", ga, lat_list, lon_list, day_rad_list,
                                                                           generated_var_list, measured_var_list)
        efficency = compute_efficency(generated_var_list_s_ga, measured_var_list_s_ga, variance)

        axs[1, 1].plot(efficency.detach().numpy(), depth, color=dict_color[ga], label=ga)
        axs[1, 1].invert_yaxis()

    axs[0, 0].set_title(list(dict_season.keys())[0])
    axs[0, 1].set_title(list(dict_season.keys())[1])
    axs[1, 0].set_title(list(dict_season.keys())[2])
    axs[1, 1].set_title(list(dict_season.keys())[3])

    for ax in axs.flat:
        ax.set(xlabel=f"{variable} ({dict_unit_measure[variable]})", ylabel='depth')
        # ax.set_xticks(range(1, len(list(dict_ga.keys())) + 1), list(dict_ga.keys()))

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize="5")

    # plt.savefig(f"{path_analysis}variance_comparison_{mode}_{epoch_model}.png")

    plt.show()
    plt.close()

    return
