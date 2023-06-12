from preprocess import preprocessing
import read_data

from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel, AutoModelForSequenceClassification


# 输入：从表中提取出的特征向量
# 输出：各个类的置信度
# 多类的分类任务

if __name__ == '__main__':
    # root_dir = "./data/DBP-Datasets"
    # train_file = "train.csv"
    # train_dataset = read_data.MyData(root_dir, train_file)
    # for idx in range(train_dataset.__len__()):
    #     print(train_dataset[idx])
        # preprocessing.preprocess(train_dataset[idx])

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = BertModel.from_pretrained("bert-base-multilingual-cased")
    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    print(encoded_input)
    print(output.last_hidden_state.shape)







