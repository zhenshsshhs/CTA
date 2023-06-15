import pandas as pd


def add_label_id():
    labels = pd.read_csv('../data/DBP-Datasets/labels.csv')
    labels = dict(zip(labels['label'], labels['label_id']))

    datas = pd.read_csv('../data/DBP-Datasets/validation.csv')
    for row in range(datas.shape[0]):
        label = datas.loc[row]["label"]
        datas.loc[[row], "label_id"] = labels[label]
        print(row)

    datas.to_csv('../data/DBP-Datasets/validation.csv', index=False)


add_label_id()
