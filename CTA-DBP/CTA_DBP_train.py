import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader
from load_data import *
from CTA_DBP_model import *

# 设置device
device = torch.device("cuda")

# 设置预训练模型
# checkpoint = 'distilbert-base-multilingual-cased'
checkpoint = 'bert-base-multilingual-cased'

# 加载数据集
train_dataset = Dataset('../data/DBP-Datasets/preprocess-train.json')
validation_dataset = Dataset('../data/DBP-Datasets/preprocess-validation.json')
# print(len(dataset))
# print(train_dataset[0])

# 加载分词器
tokenizer = BertTokenizer.from_pretrained(checkpoint)


# 数据批处理
def collate_fn(data):
    # data = data.to(device)
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]

    # 编码
    data = tokenizer.batch_encode_plus(batch_text_or_text_pairs=sents,
                                       truncation=True,
                                       padding='max_length',
                                       max_length=500,
                                       return_tensors='pt',
                                       return_length=True)
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']
    labels = torch.LongTensor(labels)

    return input_ids, attention_mask, token_type_ids, labels


# 加载数据集
train_loader = DataLoader(dataset=train_dataset, batch_size=46, collate_fn=collate_fn, shuffle=True, drop_last=True)
validation_loader = DataLoader(dataset=validation_dataset, batch_size=46, collate_fn=collate_fn, shuffle=False, drop_last=True)

# 加载预训练模型
pretrained = BertModel.from_pretrained(checkpoint)
# pretrained.to(device)

# 预训练模型不需要计算梯度
for param in pretrained.parameters():
    param.requires_grad_(False)

# 加载模型
print("loading model")
# model = CTADBPModel(pretrained)
model = torch.load('CTA-DBP-9-epoch.pth')
model = model.to(device)
print("load model success")

# test model
# print(model)

# 学习率
learning_rate = 1e-5
# 优化器
optimizer = AdamW(model.parameters(), lr=learning_rate)
# 损失函数
loss_fn = torch.nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

total_train_step = 0
epoch = 10

writer = SummaryWriter("train_logs")

for i in range(epoch):
    print("------第 {} 轮训练开始-------".format(i+1))

    model.train()

    for j, (input_ids, attention_mask, token_type_ids, labels) in enumerate(train_loader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        labels = labels.to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        out = out.to(device)
        loss = loss_fn(out, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_train_step += 1
        # print("训练次数：{}, Loss:{}".format(total_train_step, loss.item()))
        if total_train_step % 100 == 0:
            out = out.argmax(dim=1)
            accuracy = (out == labels).sum().item() / len(labels)
            print("训练次数：{}, Loss:{}, accuracy:{}".format(total_train_step, loss.item(), accuracy))
            writer.add_scalar("train_loss", loss.item(), total_train_step+18700)

        # if j % 5 == 0:
        #     out = out.argmax(dim=1)
        #     accuracy = (out == labels).sum().item() / len(labels)
        #     print(j, loss.item(), accuracy)

        # if i == 500:
        #     torch.save(model, 'CTA-SCH-Table_name-Column_index-500batch.pth')

    torch.save(model, 'CTA-DBP-{}-epoch.pth'.format(i+10))

    model.eval()
    total_test_loss = 0
    total_accuracy = 0
    total_test_step = 0
    with torch.no_grad():
        for k, (input_ids, attention_mask, token_type_ids, labels) in enumerate(validation_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            labels = labels.to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            out = out.to(device)
            loss = loss_fn(out, labels)

            out = out.argmax(dim=1)
            accuracy = (out == labels).sum().item() / len(labels)
            total_accuracy += accuracy
            total_test_loss += loss.item()
            total_test_step += 1

    print("{}-epoch validation Loss: {}, accuracy: {}".format(i, total_test_loss/total_test_step, total_accuracy/total_test_step))
    writer.add_scalar("test_accuracy", total_accuracy/total_test_step, i+10)

writer.close()


