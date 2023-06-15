import os.path
import pandas as pd
from torch.utils.data import Dataset


class Dataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.dataset = pd.read_json(self.file_path, lines=True)
        self.dataset = self.dataset.to_dict(orient='records')

    def __getitem__(self, idx):
        table_name = self.dataset[idx]['table_name'].split('_', 1)[0]
        data = table_name + ':' + self.dataset[idx]['column_data']
        label = self.dataset[idx]['label_id']
        return data, label

    def __len__(self):
        return len(self.dataset)


# test
# dataset = Dataset('data/DBP-Datasets/preprocess-train.json')
# print(len(dataset))
# print(dataset[0])

