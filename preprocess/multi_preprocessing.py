
import multiprocessing
import os
import re

import pandas as pd
import ftfy
from autocorrect import Speller
from joblib import Parallel, delayed
import multiprocessing


'''
数据预处理
将训练集处理成模型对应的输入和输出，并存在文件中
table_name, subject_data, target_data, label, label_id
'''

table_dir = "../data/Tables"


def clean_data(table_annotation, row):
    table_name = table_annotation["table_name"]
    column_index = table_annotation["column_index"]
    label = table_annotation["label"]
    label_id = table_annotation["label_id"]
    # load data
    table_path = os.path.join(table_dir, table_name)
    table_df = pd.read_json(table_path, compression='gzip', lines=True)
    column_df = table_df[column_index]
    # clean data
    column_df = column_df.fillna(' ')
    column_df = column_df.map(lambda x: str(x))
    # 删除特殊字符
    column_df = column_df.map(lambda x: re.sub('[!?✓¿|↠☆×½¼¶Þæ»#,{}]', '', x))
    # ftfy修复编码问题
    column_df = column_df.map(lambda x: ftfy.fix_text(x))
    # autocorrect修复拼写错误
    correct = Speller()
    column_df = column_df.map(lambda x: correct.autocorrect_sentence(x))
    # 将一列中的单元格值拼接起来
    column_data = '|'.join(column_df.tolist())
    column_data = column_data[:600]
    # save
    data = pd.DataFrame({'table_name': [table_name],
                         'column_index': [column_index],
                         'column_data': [column_data],
                         'label': [label],
                         'label_id': [label_id]})
    data.to_json('../data/SCH-Datasets/preprocess-validation.json', mode='a', orient='records', lines=True)
    print(row)


if __name__ == '__main__':
    table_list = os.listdir(table_dir)
    train_annotations = pd.read_csv('../data/SCH-Datasets/validation.csv')
    print(train_annotations.shape[0])

    # 并行化
    num_cores = 4
    Parallel(n_jobs=num_cores)(delayed(clean_data)(train_annotations.loc[row], row) for row in range(train_annotations.shape[0]))




