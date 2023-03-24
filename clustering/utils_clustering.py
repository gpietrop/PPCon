import os

import pandas as pd


def make_ds(training_folder):
    if training_folder == "SUPERFLOAT":
        from make_ds_clustering import make_pandas_df

        if not os.path.exists(os.getcwd() + f"/../ds/clustering/"):
            os.mkdir(os.getcwd() + f"/../ds/clustering/")

        if not os.path.exists(os.getcwd() + f"/../ds/clustering/ds_sf_clustering.csv"):
            print("making ds...")
            make_pandas_df(os.getcwd() + '/../ds/SUPERFLOAT/Float_Index.txt')
            print("superfloat clustering complete ds created")

    return
