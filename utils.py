import os


def make_ds(training_folder, flag_complete=1, flag_toy=1):
    if training_folder == "SUPERFLOAT":
        from make_ds.make_superfloat_ds import make_pandas_df, make_pandas_toy_df
        if flag_complete and not os.path.exists(os.getcwd() + "/ds/float_ds_sf.csv"):
            make_pandas_df(os.getcwd() + '/ds/SUPERFLOAT/Float_Index.txt')
            print("superfloat complete ds created")

        if flag_toy and not os.path.exists(os.getcwd() + "/ds/toy_ds_sf.csv"):
            make_pandas_toy_df(os.getcwd() + '/ds/SUPERFLOAT/Float_Index.txt')
            print("superfloat toy ds created")

    if training_folder == "CORIOLIS":
        from make_ds.make_coriolis_ds import make_pandas_df, make_pandas_toy_df

        if flag_complete and not os.path.exists(os.getcwd() + "/ds/float_ds.csv"):
            make_pandas_df(os.getcwd() + '/ds/CORIOLIS/Float_Index.txt')
            print("coriolis complete ds created")

        if flag_toy and not os.path.exists(os.getcwd() + "/ds/toy_ds.csv"):
            make_pandas_toy_df(os.getcwd() + '/ds/CORIOLIS/Float_Index.txt')
            print("coriolis toy ds created")

    return
