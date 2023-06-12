
import multiprocessing
import os
import re

import pandas as pd
import ftfy
import spacy
from autocorrect import Speller
from joblib import Parallel, delayed
import multiprocessing


'''
数据预处理
将训练集处理成模型对应的输入和输出，并存在文件中
table_name, subject_data, target_data, label
'''

table_dir = "../data/Tables"


def clean_data(table_name, column_index, label, row):
    print(table_name, row)
    table_path = os.path.join(table_dir, table_name)
    table_df = pd.read_json(table_path, compression='gzip', lines=True)
    # 获取目标列的数据
    column_df = table_df[column_index]
    column_df = column_df.fillna(' ')
    column_df = column_df.map(lambda x: str(x))
    # 删除特殊字符
    column_df = column_df.map(lambda x: re.sub('[!?✓¿|↠☆×½¼¶Þæ»#\"\'\t\n\r,，\[\]{}]', '', x))
    # ftfy修复编码问题
    column_df = column_df.map(lambda x: ftfy.fix_text(x))
    # autocorrect修复拼写错误
    correct = Speller()
    column_df = column_df.map(lambda x: correct.autocorrect_sentence(x))
    # 将一列中的单元格值拼接起来
    column_data = ' | '.join(column_df.tolist())
    column_data = column_data[:600]
    column_data ="\"" + column_data + "\""

    # save
    data = pd.DataFrame({'table_name': [table_name], 'column_index': [column_index], 'column_data': [column_data], 'label': [label]})
    data.to_csv('../data/SCH-Datasets/preprocess-train.csv', mode='a', index=0, header=0)
    # return column_data


if __name__ == '__main__':
    table_list = os.listdir(table_dir)
    # print(len(table_list))
    # print(table_list[0])
    train_annotations = pd.read_csv('../data/SCH-Datasets/train.csv')
    # for row in range(train_annotations.shape[0]):
    #     table_name = train_annotations.loc[row]["table_name"]
    #     column_index = train_annotations.loc[row]["column_index"]
    #     label = train_annotations.loc[row]["label"]
    #     column_data = clean_data(table_name, column_index)
    #     data = pd.DataFrame({'table_name': [table_name], 'column_data': [column_data], 'label': [label]})
    #     data.to_csv('../data/SCH-Datasets/preprocess-train.csv', mode='a', index=0, header=0)
    #     print(row)
    #     print(table_name, column_index, column_data, label)



    # 并行化
    num_cores = 10
    Parallel(n_jobs=num_cores)(delayed(clean_data)(train_annotations.loc[row]["table_name"],
                                                   train_annotations.loc[row]["column_index"],
                                                   train_annotations.loc[row]["label"], row)
                               for row in range(train_annotations.shape[0]))


