from utils_analysis import *


def reconstruction_profiles(variable, date_model, epoch_model, mode):
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
        plt.figure(figsize=(6, 7))

        plt.plot(measured_profile.detach().numpy(), depth, lw=3,
                 color="#2CA02C", label=f"Measured")
        plt.plot(generated_profile.detach().numpy(), depth, lw=3, linestyle=(0, (3, 1, 1, 1)),
                 color="#1F77B4", label=f"PPCon")
        plt.gca().invert_yaxis()

        plt.xlabel(f"{variable} [{dict_unit_measure[variable]}]")
        plt.ylabel(r"depth [$m$]")

        plt.legend()
        plt.savefig(f"{path_analysis}profile_{round(lat, 2)}_{round(lon, 2)}.png")
        plt.close()

    return


def reconstruction_profile_MLP(variable, season, date_model, epoch_model, mode):
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

        plt.figure(figsize=(6, 7))

        plt.plot(torch.squeeze(measured_var)[cut_sup:-cut_inf], depth[cut_sup:-cut_inf], lw=3, color="#2CA02C",
                 label="Measured")
        plt.plot(torch.squeeze(generated_gloria_var)[cut_sup:-cut_inf], depth[cut_sup:-cut_inf], lw=3,
                 color="#BCBD22", linestyle=(0, (5, 1)), label="MLP")
        plt.plot(torch.squeeze(generated_var)[cut_sup:-cut_inf], depth[cut_sup:-cut_inf], lw=3,
                 color="#1F77B4", linestyle=(0, (3, 1, 1, 1)), label="PPCon")
        plt.gca().invert_yaxis()

        plt.xlabel(f"{variable} [{dict_unit_measure[variable]}]")
        plt.ylabel(r"depth [$m$]")

        plt.legend()
        plt.savefig(f"{path_analysis}/method_comparison_{round(lat.item(), 2)}_{round(lon.item(), 2)}.png")
        plt.close()

    return