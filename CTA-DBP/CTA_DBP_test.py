import pandas as pd
import torch
from transformers import BertTokenizer

# 设置device
device = torch.device("cuda")

# load model
model = torch.load("CTA-DBP-best.pth")
model = model.to(device)
print('load model success')

# load labels
labels = pd.read_csv('../data/SCH-Datasets/labels.csv')
labels = dict(zip(labels['label_id'], labels['label']))
print('load labels success')

# load data
test_path = "../data/DBP-Datasets/preprocess-validation.json"
test_data = pd.read_json(test_path, lines=True)
test_data = test_data.to_dict(orient='records')
print('load data success')

print('start annotate')
for row in range(test_data.shape[0]):
    column_index = test_data[row]['column_index']
    table_name = test_data[row]['table_name']
    data = table_name.split('_', 1)[row] + ' : ' + test_data[row]['column_data']

    # 编码
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    data = tokenizer(data, truncation=True, max_length=500, padding='max_length', return_tensors='pt')
    input_ids = data['input_ids']
    input_ids = input_ids.to(device)
    attention_mask = data['attention_mask']
    attention_mask = attention_mask.to(device)
    token_type_ids = data['token_type_ids']
    token_type_ids = token_type_ids.to(device)

    # use model predict
    model.eval()
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # out = out.to(device)

    print(out)
    out = out.argmax(dim=1)
    print(out)
    out = out.item()
    print(out)
    # id to label
    label = labels[out]
    print(table_name, column_index, label)

    # save annotation
    annotation = pd.DataFrame({'table_name': [table_name], 'column_index': [column_index], 'label': [label]})
    annotation.to_csv('../data/DBP-Datasets/annotate-validation.csv', mode='a', index=0, header=0)

    print(row)

print("done")




