from process_data import preprocessing
import read_data

from torch.utils.data import DataLoader


# 输入：主语列的信息和目标列的信息
# 输出：各个类的概率
# 多类的分类任务

if __name__ == '__main__':
    root_dir = "./data/DBP-Datasets"
    train_file = "train.csv"
    train_dataset = read_data.MyData(root_dir, train_file)
    for idx in range(train_dataset.__len__()):
        print(train_dataset[idx])
        # preprocessing.preprocess(train_dataset[idx])





