import read_data
from torch.utils.data import DataLoader


if __name__ == '__main__':
    root_dir = "./data/SCH-Datasets"
    train_file = "train.csv"
    train_dataset = read_data.MyData(root_dir, train_file)
    train_loader = DataLoader(dataset=train_dataset)
    # for idx in range(train_dataset.__len__()):
    #     print(train_dataset[idx])
    # for data in train_loader:
    #     print(data)
    print(train_dataset[0])

