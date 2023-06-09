
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


def fix_data(table_name):
    print(table_name)
    process_table_dir = "../data/PreprocessTables"
    table_path = os.path.join(table_dir, table_name)
    table_df = pd.read_json(table_path, compression='gzip', lines=True)
    table_df = table_df.fillna("")
    table_df = table_df.applymap(lambda x: str(x))
    # 删除特殊字符
    table_df = table_df.applymap(lambda x: re.sub('[!?✓¿|↠☆×½¼¶Þæ»#]', '', x))
    # ftfy修复编码问题
    clean_df = table_df.applymap(lambda x: ftfy.fix_text(x))
    # autocorrect修复拼写错误
    correct = Speller()
    correct_df = clean_df.applymap(lambda x: correct.autocorrect_sentence(x))
    # 保存表格
    # process_table_path = os.path.join(process_table_dir, table_name)
    # correct_df.to_json(process_table_path, compression='gzip', orient='records', lines=True)


    # print(table_df)
    # print(correct_df)
    # for row in range(table_df.shape[0]):
    #     for col in range(table_df.shape[1]):
    #         table_value = table_df.loc[row][col]
    #         clean_value = clean_df.loc[row][col]
    #         correct_value = correct_df.loc[row][col]
    #         if table_value != clean_value and clean_value != correct_value:
    #             print("table_value: {}, clean_value: {}, correct_value: {}".format(table_value, clean_value, correct_value))


if __name__ == '__main__':
    table_list = os.listdir(table_dir)
    print(len(table_list))
    print(table_list[0])
    train_annotations = pd.read_csv('../data/SCH-Datasets/train.csv')
    for row in range(train_annotations.shape[0]):
        print(train_annotations.loc[row])

    # 并行化主语列预测
    # num_cores = 8
    # Parallel(n_jobs=num_cores)(delayed(predict_subject)(table_list[idx], idx) for idx in range(len(table_list)))


