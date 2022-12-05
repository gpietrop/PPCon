import pandas as pd
import torch
from torch.utils.data import Dataset


def from_string_to_tensor(string):
    # string = df.iloc[6, 1][8:-2]
    string = string.split(",")
    out = torch.zeros(200)
    for ind in range(len(string)):
        out[ind] = torch.tensor(float(string[ind]))
    return out


class FloatDataset(Dataset):

    def __init__(self, path_df=None):
        super().__init__()
        if path_df is not None:
            self.path_df = path_df
            self.df = pd.read_csv(self.path_df)
        else:
            raise Exception("Paths should be given as input to initialize the Float class.")

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.df.iloc[0, :])

    def __getitem__(self, index):
        """Generates one sample of data"""
        self.samples = self.df.iloc[:, index + 1].tolist()  # Select sample

        year = torch.tensor(self.samples[0])
        day_rad = torch.tensor(self.samples[1])
        lat = torch.tensor(self.samples[2])
        lon = torch.tensor(self.samples[3])
        temp = from_string_to_tensor(self.samples[4])
        psal = from_string_to_tensor(self.samples[5])
        doxy = from_string_to_tensor(self.samples[6])

        return year, day_rad, lat, lon, temp, psal, doxy
