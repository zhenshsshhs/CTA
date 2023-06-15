import pandas as pd
import torch
from transformers import BertTokenizer

# 设置device
device = torch.device("cuda")

# load data
test_path = "../data/SCH-Datasets/preprocess-validation.json"
test_data = pd.read_json(test_path, lines=True)
test_data = test_data.to_dict(orient='records')
column_index = test_data[0]['column_index']
table_name = test_data[0]['table_name']
data = table_name.split('_', 1)[0] + ' : ' + test_data[0]['column_data']
print(data)

# load labels
labels = pd.read_csv('../data/SCH-Datasets/labels.csv')
labels = dict(zip(labels['label_id'], labels['label']))

# 编码
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
data = tokenizer(data, truncation=True, max_length=500, padding='max_length', return_tensors='pt')
input_ids = data['input_ids']
input_ids = input_ids.to(device)
print(input_ids.is_cuda)
attention_mask = data['attention_mask']
attention_mask = attention_mask.to(device)
token_type_ids = data['token_type_ids']
token_type_ids = token_type_ids.to(device)

# load model
model = torch.load("CTA-SCH-0-epoch.pth")
model = model.to(device)
# print(model)
model.eval()
with torch.no_grad():
    out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    # out = out.to(device)

print(out)
out = out.argmax(dim=1)
print(out)
out = out.item()
print(out)
label = labels[out]
print(table_name, column_index, label)


