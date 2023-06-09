import os

import pandas as pd
import spacy

table_dir = "../data/Tables"

# 主语列预测


# 判断单元格值是否是文本类型
def is_numeric(doc):
    for ent in doc.ents:
        # print(ent.text, ent.label_)
        tag = ent.label_
        if tag in ['DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']:
            return True
        else:
            return False


# 预测表格的主语列，并将主语列索引和表格名保存到文件中
def predict_subject(table_name, idx):
    print(table_name, idx)
    table_path = os.path.join(table_dir, table_name)
    table_df = pd.read_json(table_path, compression='gzip', lines=True)
    table_df = table_df.fillna("")
    table_df = table_df.applymap(lambda x: str(x))
    # print(table_df)
    nlp = spacy.load("en_core_web_sm")
    subject_index = 0  # 默认为第一列
    for col in range(table_df.shape[1]):
        literal = 0
        named_entity = 0
        row_num = table_df.shape[0]
        for row in range(row_num):
            cell_value = table_df.loc[row][col]
            doc = nlp(cell_value)
            if is_numeric(doc):
                literal += 1
            else:
                named_entity += 1
                if named_entity > row_num / 2:
                    break
        # print("na:{}, li:{}".format(named_entity, literal))
        # 命名实体的标签多于文本类型的标签，则认为该列是命名实体列，选择第一个命名实体列作为主语列
        if named_entity > literal:
            subject_index = col
            break
    # 把表名和主语列存入文件中
    data = pd.DataFrame({'table_name': [table_name], 'subject_index': [subject_index]})
    data.to_csv('../data/table_subject.csv', mode='a', index=0, header=0)


if __name__ == '__main__':
    table_list = os.listdir(table_dir)
    print(len(table_list))
    for idx in range(len(table_list)):
        table_name = table_list[idx]
        predict_subject(table_name, idx)
