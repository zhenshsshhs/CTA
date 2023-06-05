import os.path
import pandas as pd
import torch

from torch.utils.data import Dataset


class MyData(Dataset):

    def __init__(self, root_dir, file_name):
        self.root_dir = root_dir
        self.file_name = file_name
        self.file_path = os.path.join(self.root_dir, file_name)

    def __getitem__(self, idx):
        dataset = pd.read_csv(self.file_path)
        table_name = dataset.loc[idx]['table_name']
        column_index = dataset.loc[idx]['column_index']
        label = dataset.loc[idx]['label']
        if dataset.shape[1] == 3:
            label = dataset.loc[idx]['label']
            return table_name, column_index, label
        return table_name, column_index
        # table_dir = "./data/Tables"
        # table_path = os.path.join(table_dir, table_name)
        # table_df = pd.read_json(table_path, compression='gzip', lines=True)
        # column_data = table_df[column_index]
        # return column_data, label

    def __len__(self):
        return pd.read_csv(self.file_path).shape[0]
