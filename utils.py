import os
import random

import pandas as pd


def shuffle_dict(my_dict):
    items = list(my_dict.items())  # List of tuples of (key,values)
    random.shuffle(items)
    return dict(items)


def make_ds(training_folder, variable, flag_complete=1, flag_toy=1):
    if training_folder == "SUPERFLOAT":
        from make_ds.make_superfloat_ds import make_pandas_df, make_pandas_toy_df

        if not os.path.exists(os.getcwd() + f"/ds/{variable}/"):
            os.mkdir(os.getcwd() + f"/ds/{variable}/")

        if flag_complete and not os.path.exists(os.getcwd() + f"/ds/{variable}/float_ds_sf_train.csv"):
            print("making ds...")
            make_pandas_df(os.getcwd() + '/ds/SUPERFLOAT/Float_Index.txt', variable=variable)
            print("superfloat complete ds created")

        if flag_toy and not os.path.exists(os.getcwd() + f"/ds/{variable}/toy_ds_sf.csv"):
            print("making ds...")
            # make_pandas_toy_df(os.getcwd() + '/ds/SUPERFLOAT/Float_Index.txt', variable=variable)
            print("superfloat toy ds created")

    if training_folder == "CORIOLIS":
        from make_ds.make_coriolis_ds import make_pandas_df, make_pandas_toy_df

        if flag_complete and not os.path.exists(os.getcwd() + f"{variable}/ds/float_ds.csv"):
            make_pandas_df(os.getcwd() + '/ds/CORIOLIS/Float_Index.txt')
            print("coriolis complete ds created")

        if flag_toy and not os.path.exists(os.getcwd() + f"{variable}/ds/toy_ds.csv"):
            make_pandas_toy_df(os.getcwd() + '/ds/CORIOLIS/Float_Index.txt')
            print("coriolis toy ds created")

    return


def save_ds_info(training_folder, flag_toy, batch_size, epochs, lr, dp_rate, lambda_l2_reg, save_dir, alpha_smooth_reg):
    dict_info = {'train_ds': [training_folder],
                 'is_toy': [flag_toy],
                 'batch_size': [batch_size],
                 'epoch': [epochs],
                 'lr': [lr],
                 'dp_rate': [dp_rate],
                 'lambda_l2_reg': [lambda_l2_reg],
                 'alpha_smooth_reg': alpha_smooth_reg}
    pd_ds = pd.DataFrame(dict_info)
    pd_ds.to_csv(save_dir + '/info.csv')

    return
