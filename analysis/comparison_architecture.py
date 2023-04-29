from analysis_utils import *


def get_reconstruction_comparison(variable, season, date_model, epoch_model, mode):
    # Create savedir
    path_analysis = os.getcwd() + f"/../results/NITRATE/{date_model}/fig/comp"
    if not os.path.exists(path_analysis):
        os.mkdir(path_analysis)
    # Upload the input ds

    dict_season = {'W': [0, 91], 'SP': [92, 182], 'SU': [183, 273], 'A': [274, 365]}

    path_float = f"/home/gpietropolli/Desktop/canyon-float/ds/NITRATE/float_ds_sf_{mode}.csv"
    if mode == "all":
        path_float = f"/home/gpietropolli/Desktop/canyon-float/ds/NITRATE/float_ds_sf.csv"
    dataset = FloatDataset(path_float)
    ds = DataLoader(dataset, shuffle=True)

    dir_model = os.getcwd() + f"/../results/NITRATE/{date_model}/model"
    info = pd.read_csv(os.getcwd() + f"/../results/NITRATE/{date_model}/info.csv")

    # Upload and evaluate the model
    model_day, model_year, model_lat, model_lon, model = upload_and_evaluate_model(dir_model=dir_model, info_model=info,
                                                                                   ep=epoch_model)

    lat_list = list()
    lon_list = list()
    day_rad_list = list()
    measured_var_list = list()
    generated_var_list = list()
    suazade_var_list = list()
    gloria_var_list = list()
    number_seasonal_sample = 0

    sum_generated = torch.zeros(1, 1, 200)
    sum_measured = torch.zeros(1, 1, 200)
    sum_gloria = torch.zeros(1, 1, 200)
    sum_suazade = torch.zeros(200)

    for sample in ds:
        year, day_rad, lat, lon, temp, psal, doxy, measured_var = sample
        day_sample = from_day_rad_to_day(day_rad=day_rad)
        if season != "all" and not dict_season[season][0] <= day_sample <= dict_season[season][
            1] and random.random() < 0.2:
            continue
        number_seasonal_sample += 1
        generated_suazade_var = get_suazade_profile(year, day_rad, lat, lon, torch.squeeze(temp), torch.squeeze(psal),
                                                    torch.squeeze(doxy), torch.squeeze(measured_var))

        suazade_var_list.append(generated_suazade_var)
        sum_suazade += generated_suazade_var

        generated_gloria_var = get_gloria_profile(year, day_rad, lat, lon, torch.squeeze(temp), torch.squeeze(psal),
                                                  torch.squeeze(doxy), torch.squeeze(measured_var))
        gloria_var_list.append(generated_gloria_var)
        sum_gloria += generated_gloria_var

        output_day = model_day(day_rad.unsqueeze(1))
        output_year = model_year(year.unsqueeze(1))
        output_lat = model_lat(lat.unsqueeze(1))
        output_lon = model_lon(lon.unsqueeze(1))

        output_day = torch.transpose(output_day.unsqueeze(0), 0, 1)
        output_year = torch.transpose(output_year.unsqueeze(0), 0, 1)
        output_lat = torch.transpose(output_lat.unsqueeze(0), 0, 1)
        output_lon = torch.transpose(output_lon.unsqueeze(0), 0, 1)
        temp = torch.transpose(temp.unsqueeze(0), 0, 1)
        psal = torch.transpose(psal.unsqueeze(0), 0, 1)
        doxy = torch.transpose(doxy.unsqueeze(0), 0, 1)

        x = torch.cat((output_day, output_year, output_lat, output_lon, temp, psal, doxy), 1)
        generated_var = model(x.float())
        generated_var = generated_var.detach()  # torch.squeeze????

        lat_list.append(lat.item())
        lon_list.append(lon.item())
        day_rad_list.append(day_rad.item())

        generated_var_list.append(generated_var)
        sum_generated += generated_var
        measured_var_list.append(measured_var)
        sum_measured += measured_var

        depth = np.linspace(0, dict_max_pressure["NITRATE"], len(torch.squeeze(generated_var)))

        cut_sup = 3
        cut_inf = 10

        plt.plot(torch.squeeze(measured_var)[cut_sup:-cut_inf], depth[cut_sup:-cut_inf], label="measured")
        plt.plot(generated_suazade_var[cut_sup:-cut_inf], depth[cut_sup:-cut_inf], label="MLP Fourrier")
        plt.plot(torch.squeeze(generated_gloria_var)[cut_sup:-cut_inf], depth[cut_sup:-cut_inf],
                 label="MLP Pietropolli")
        plt.plot(torch.squeeze(generated_var)[cut_sup:-cut_inf], depth[cut_sup:-cut_inf], label="convolutional")
        plt.gca().invert_yaxis()

        plt.xlabel(f"{variable} {dict_unit_measure[variable]}")
        plt.ylabel('depth')

        plt.legend()
        plt.savefig(f"{path_analysis}/method_comparison_{round(lat.item(), 2)}_{round(lon.item(), 2)}.png")
        plt.close()

    generated_mean = sum_generated / number_seasonal_samples
    measured_mean = sum_measured / number_seasonal_samples
    gloria_mean = sum_gloria / number_seasonal_samples
    suazade_mean = sum_suazade / number_seasonal_samples

    max_pres = dict_max_pressure[variable]
    depth = np.linspace(0, max_pres, len(generated_profile.detach().numpy()))
    plt.plot(torch.squeeze(measured_mean), depth, label=f"measured")
    plt.plot(suazade_mean, depth, label=f"MLP Fourrier")
    plt.plot(torch.squeeze(gloria_mean), depth, label=f"MLP Pietropolli")
    plt.plot(torch.squeeze(generated_mean), depth, label=f"convolutional")
    plt.gca().invert_yaxis()

    plt.xlabel(f"{variable} [{dict_unit_measure[variable]}]")
    plt.ylabel('depth [m]')

    plt.legend()
    plt.savefig(f"{path_analysis}/method_comparison_{season}.png")
    plt.show()
    plt.close()

    return


def get_reconstruction_comparison_presentation(variable, season, date_model, epoch_model, mode):
    # Create savedir
    path_analysis = os.getcwd() + f"/../results/NITRATE/{date_model}/fig/comp_presentation"
    if not os.path.exists(path_analysis):
        os.mkdir(path_analysis)
    # Upload the input ds

    dict_season = {'W': [0, 91], 'SP': [92, 182], 'SU': [183, 273], 'A': [274, 365]}

    path_float = f"/home/gpietropolli/Desktop/canyon-float/ds/NITRATE/float_ds_sf_{mode}.csv"
    if mode == "all":
        path_float = f"/home/gpietropolli/Desktop/canyon-float/ds/NITRATE/float_ds_sf.csv"
    dataset = FloatDataset(path_float)
    ds = DataLoader(dataset, shuffle=True)

    dir_model = os.getcwd() + f"/../results/NITRATE/{date_model}/model"
    info = pd.read_csv(os.getcwd() + f"/../results/NITRATE/{date_model}/info.csv")

    # Upload and evaluate the model
    model_day, model_year, model_lat, model_lon, model = upload_and_evaluate_model(dir_model=dir_model, info_model=info,
                                                                                   ep=epoch_model)

    lat_list = list()
    lon_list = list()
    day_rad_list = list()
    measured_var_list = list()
    generated_var_list = list()
    gloria_var_list = list()
    number_seasonal_sample = 0

    sum_generated = torch.zeros(1, 1, 200)
    sum_measured = torch.zeros(1, 1, 200)
    sum_gloria = torch.zeros(1, 1, 200)

    for sample in ds:
        year, day_rad, lat, lon, temp, psal, doxy, measured_var = sample
        day_sample = from_day_rad_to_day(day_rad=day_rad)
        if season != "all" and not dict_season[season][0] <= day_sample <= dict_season[season][
            1] and random.random() < 0.2:
            continue
        number_seasonal_sample += 1

        generated_gloria_var = get_gloria_profile(year, day_rad, lat, lon, torch.squeeze(temp), torch.squeeze(psal),
                                                  torch.squeeze(doxy), torch.squeeze(measured_var))
        gloria_var_list.append(generated_gloria_var)
        sum_gloria += generated_gloria_var

        output_day = model_day(day_rad.unsqueeze(1))
        output_year = model_year(year.unsqueeze(1))
        output_lat = model_lat(lat.unsqueeze(1))
        output_lon = model_lon(lon.unsqueeze(1))

        output_day = torch.transpose(output_day.unsqueeze(0), 0, 1)
        output_year = torch.transpose(output_year.unsqueeze(0), 0, 1)
        output_lat = torch.transpose(output_lat.unsqueeze(0), 0, 1)
        output_lon = torch.transpose(output_lon.unsqueeze(0), 0, 1)
        temp = torch.transpose(temp.unsqueeze(0), 0, 1)
        psal = torch.transpose(psal.unsqueeze(0), 0, 1)
        doxy = torch.transpose(doxy.unsqueeze(0), 0, 1)

        x = torch.cat((output_day, output_year, output_lat, output_lon, temp, psal, doxy), 1)
        generated_var = model(x.float())
        generated_var = generated_var.detach()  # torch.squeeze????

        lat_list.append(lat.item())
        lon_list.append(lon.item())
        day_rad_list.append(day_rad.item())

        generated_var_list.append(generated_var)
        sum_generated += generated_var
        measured_var_list.append(measured_var)
        sum_measured += measured_var

        depth = np.linspace(0, dict_max_pressure["NITRATE"], len(torch.squeeze(generated_var)))

        cut_sup = 3
        cut_inf = 10

        plt.plot(torch.squeeze(measured_var)[cut_sup:-cut_inf], depth[cut_sup:-cut_inf], lw=2.5, color="green",
                 label="measured")
        plt.plot(torch.squeeze(generated_gloria_var)[cut_sup:-cut_inf], depth[cut_sup:-cut_inf], lw=2.5,
                 color="darkorange", linestyle="dashed", label="MLP Pietropolli")
        plt.plot(torch.squeeze(generated_var)[cut_sup:-cut_inf], depth[cut_sup:-cut_inf], lw=2.5,
                 color="blue", linestyle="dashed", label="convolutional")
        plt.gca().invert_yaxis()

        plt.xlabel(f"{variable} [{dict_unit_measure[variable]}]")
        plt.ylabel('depth [m]')

        plt.legend()
        plt.savefig(f"{path_analysis}/method_comparison_{round(lat.item(), 2)}_{round(lon.item(), 2)}.png")
        plt.close()

    return


def get_reconstruction_comparison_gpietrop(variable, season, date_model, epoch_model, mode):
    # Create savedir
    path_analysis = os.getcwd() + f"/../results/NITRATE/{date_model}/fig/comp_gpietrop"
    if not os.path.exists(path_analysis):
        os.mkdir(path_analysis)
    # Upload the input ds

    dict_season = {'W': [0, 91], 'SP': [92, 182], 'SU': [183, 273], 'A': [274, 365]}

    path_float = f"/home/gpietropolli/Desktop/canyon-float/ds/NITRATE/float_ds_sf_{mode}.csv"
    if mode == "all":
        path_float = f"/home/gpietropolli/Desktop/canyon-float/ds/NITRATE/float_ds_sf.csv"
    dataset = FloatDataset(path_float)
    ds = DataLoader(dataset, shuffle=True)

    dir_model = os.getcwd() + f"/../results/NITRATE/{date_model}/model"
    info = pd.read_csv(os.getcwd() + f"/../results/NITRATE/{date_model}/info.csv")

    # Upload and evaluate the model
    model_day, model_year, model_lat, model_lon, model = upload_and_evaluate_model(dir_model=dir_model, info_model=info,
                                                                                   ep=epoch_model)

    lat_list = list()
    lon_list = list()
    day_rad_list = list()
    measured_var_list = list()
    gloria_var_list = list()
    number_seasonal_sample = 0

    sum_measured = torch.zeros(1, 1, 200)
    sum_gloria = torch.zeros(1, 1, 200)

    for sample in ds:
        year, day_rad, lat, lon, temp, psal, doxy, measured_var = sample
        day_sample = from_day_rad_to_day(day_rad=day_rad)
        if season != "all" and not dict_season[season][0] <= day_sample <= dict_season[season][
            1] and random.random() < 0.2:
            continue
        number_seasonal_sample += 1

        generated_gloria_var = get_gloria_profile(year, day_rad, lat, lon, torch.squeeze(temp), torch.squeeze(psal),
                                                  torch.squeeze(doxy), torch.squeeze(measured_var))
        gloria_var_list.append(generated_gloria_var)
        sum_gloria += generated_gloria_var

        lat_list.append(lat.item())
        lon_list.append(lon.item())
        day_rad_list.append(day_rad.item())

        measured_var_list.append(measured_var)
        sum_measured += measured_var

        depth = np.linspace(0, dict_max_pressure["NITRATE"], len(torch.squeeze(measured_var)))

        cut_sup = 3
        cut_inf = 10

        plt.plot(torch.squeeze(measured_var)[cut_sup:-cut_inf], depth[cut_sup:-cut_inf], lw=2.5, label="measured",
                 color='green')
        plt.plot(torch.squeeze(generated_gloria_var)[cut_sup:-cut_inf], depth[cut_sup:-cut_inf], lw=2.5,
                 linestyle="dashed", label="MLP Pietropolli", color='darkorange')
        plt.gca().invert_yaxis()

        plt.xlabel(f"{variable} [{dict_unit_measure[variable]}]")
        plt.ylabel('depth [m]')

        plt.legend()
        plt.savefig(f"{path_analysis}/method_comparison_{round(lat.item(), 2)}_{round(lon.item(), 2)}.png")
        plt.close()

    measured_mean = sum_measured / number_seasonal_samples
    gloria_mean = sum_gloria / number_seasonal_samples

    max_pres = dict_max_pressure[variable]
    depth = np.linspace(0, max_pres, len(generated_profile.detach().numpy()))
    plt.plot(torch.squeeze(measured_mean), depth, lw=3, label=f"measured")
    plt.plot(torch.squeeze(gloria_mean), depth, lw=3, label=f"MLP Pietropolli")
    plt.gca().invert_yaxis()

    plt.xlabel(f"{variable} [{dict_unit_measure[variable]}]")
    plt.ylabel('depth [m]')

    plt.legend()
    plt.savefig(f"{path_analysis}/method_comparison_{season}.png")
    plt.show()
    plt.close()

    return
