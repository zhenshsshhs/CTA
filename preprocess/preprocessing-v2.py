
import multiprocessing
import os
import re

import pandas as pd
import ftfy
from autocorrect import Speller


'''
数据预处理
将训练集处理成模型对应的输入和输出，并存在文件中
table_name, subject_data, target_data, label, label_id
'''

table_dir = "../data/Tables"


def clean_data(column_df):
    correct = Speller()
    column_df = column_df.fillna(' ')
    column_df = column_df.map(lambda x: str(x))
    count = 0
    column_data = ''
    for row in range(column_df.shape[0]):
        row_data = column_df[row]
        if row_data == ' ':
            continue
        else:
            # 删除特殊字符
            row_data = re.sub('[!?✓¿|↠☆×½¼¶Þæ»#,{}]', '', row_data)
            # ftfy修复编码问题
            row_data = ftfy.fix_text(row_data)
            # autocorrect修复拼写错误
            row_data = correct.autocorrect_sentence(row_data)
            # 拼接单元格值
            column_data = column_data + ',' + row_data
            count += 1
            # 只保留前十个有内容的单元格值
            if count == 10:
                break
            # print(row_data)
    column_data = column_data.lstrip(',')
    # print(column_data)
    return column_data


if __name__ == '__main__':
    table_list = os.listdir(table_dir)
    # print(len(table_list))
    # print(table_list[0])
    train_annotations = pd.read_csv('../data/SCH-Datasets/train.csv')
    for row in range(train_annotations.shape[0]):
        table_name = train_annotations.loc[row]["table_name"]
        column_index = train_annotations.loc[row]["column_index"]
        label = train_annotations.loc[row]["label"]
        label_id = train_annotations.loc[row]["label_id"]
        # load data
        table_path = os.path.join(table_dir, table_name)
        table_df = pd.read_json(table_path, compression='gzip', lines=True)
        column_df = table_df[column_index]
        # clean data
        column_data = clean_data(column_df)
        # save
        data = pd.DataFrame({'table_name': [table_name], 'column_index': [column_index], 'column_data': [column_data], 'label': [label], 'label_id': [label_id]})
        data.to_json('../data/SCH-Datasets/preprocess-train.json', mode='a', orient='records', lines=True)
        print(row)




