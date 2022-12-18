import os


def make_ds(training_folder, flag_complete=1, flag_toy=1):
    if training_folder == "SUPERFLOAT":
        from make_ds.make_superfloat_ds import make_pandas_df, make_pandas_toy_df
        if flag_complete:
            make_pandas_df(os.getcwd() + '/ds/SUPERFLOAT/Float_Index.txt')
        if flag_toy:
            make_pandas_toy_df(os.getcwd() + '/ds/SUPERFLOAT/Float_Index.txt')

    if training_folder == "CORIOLIS":
        from make_ds.make_coriolis_ds import make_pandas_df, make_pandas_toy_df

        if flag_complete:
            make_pandas_df(os.getcwd() + '/ds/CORIOLIS/Float_Index.txt')
        if flag_toy:
            make_pandas_toy_df(os.getcwd() + '/ds/CORIOLIS/Float_Index.txt')
    return
