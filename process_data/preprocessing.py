
# 数据预处理
# 使用ftfy修复编码问题
import multiprocessing
import os
import re

import pandas as pd
import ftfy
import spacy
from autocorrect import Speller
from joblib import Parallel, delayed
import multiprocessing


def is_numeric(doc):
    for ent in doc.ents:
        # print(ent.text, ent.label_)
        tag = ent.label_
        if tag in ['DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']:
            return True
        else:
            return False


def predict_subject(table_data):
    # print(table_data)
    nlp = spacy.load("en_core_web_sm")
    subject_index = 0
    for col in range(table_data.shape[1]):
        literal = 0
        named_entity = 0
        for row in range(table_data.shape[0]):
            cell_value = table_data.loc[row][col]
            doc = nlp(cell_value)
            if is_numeric(doc):
                literal += 1
            else:
                named_entity += 1
        # print("na:{}, li:{}".format(named_entity, literal))
        if named_entity > literal:
            subject_index = col
            break
    return subject_index


def preprocess(table_name):
    print(table_name)
    table_dir = "../data/Tables"
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

    subject_column = predict_subject(correct_df)
    print(subject_column)

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
    table_dir = os.listdir('../data/Tables')
    # for idx in range(len(table_dir)):
    #     table_name = table_dir[idx]
    #     preprocess(table_name)
    #     print(table_name)

    preprocess(table_dir[0])

    # 并行化预处理
    # num_cores = 8
    # Parallel(n_jobs=num_cores)(delayed(preprocess)(table_dir[idx]) for idx in range(len(table_dir)))

