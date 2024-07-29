from utils_analysis import *


def moving_average(data, window_size):
    if window_size % 2 == 0:
        window_size += 1  # Ensure window size is odd for symmetry
    pad_size = window_size // 2
    padded_data = np.pad(data, pad_size, mode='edge')
    cumsum_vec = np.cumsum(np.insert(padded_data, 0, 0))
    return (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size


def reconstruction_profiles(variable, date_model, epoch_model, mode):
    path_analysis = os.getcwd() + f"/../results/{variable}/{date_model}/fig/"
    if not os.path.exists(path_analysis):
        os.mkdir(path_analysis)
    path_analysis = os.getcwd() + f"/../results/{variable}/{date_model}/fig/profile_{mode}_{epoch_model}/"
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
        plt.figure(figsize=(6, 7))

        measured_profile = moving_average(measured_profile.detach().numpy(), 3)
        generated_profile = moving_average(generated_profile.detach().numpy(), 3)
        # generated_profile = moving_average(generated_profile.detach().numpy(), 3)

        plt.plot(measured_profile, depth, lw=3, color="#2CA02C", label=f"Measured")
        plt.plot(generated_profile, depth, lw=3, linestyle=(0, (3, 1, 1, 1)), color="#1F77B4", label=f"PPCon")
        plt.gca().invert_yaxis()

        plt.xlabel(f"{dict_var_name[variable]} [{dict_unit_measure[variable]}]")
        plt.ylabel(r"Depth [$m$]")

        if variable == "BBP700":
            ax = plt.gca()
            x_labels = ax.get_xticks()
            ax.set_xticklabels(['{:,.0e}'.format(x) for x in x_labels])

        plt.legend()
        plt.tight_layout()

        plt.legend()
        plt.savefig(f"{path_analysis}profile_{round(lat, 2)}_{round(lon, 2)}.png") #, dpi=1200)
        plt.close()

    return