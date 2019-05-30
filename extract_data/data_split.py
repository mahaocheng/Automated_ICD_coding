import pandas as pd


filter_num = 100    # 过滤参数，将病例数小于filter_num的ICD编码去掉
valid_rate = 0.1    # 验证集比例
test_rate = 0.1     # 测试集比例
entry_names = ['主诉', '现病史', '检查报告', '首次病程记录', '查房记录', '出院记录']
label_name = ['ICD编码']


def data_filter(data, filter_num):
    ICD_dict = dict(data["ICD编码"].value_counts())
    reserved_ICD = []
    for code in ICD_dict.keys():
        if ICD_dict[code] >= filter_num:
            reserved_ICD.append(code)
    droped_index = []
    for i, code in enumerate(data["ICD编码"]):
        if code not in reserved_ICD:
            droped_index.append(i)
    data = data.drop(labels=droped_index, axis=0).reset_index(drop=True)
    return data

def data_split(data, valid_rate, test_rate):
    shuffle_data = data.sample(frac=1).reset_index(drop=True)
    valid_size = int(len(shuffle_data)*valid_rate)
    test_size = int(len(shuffle_data)*test_rate)
    train_size = len(shuffle_data) - valid_size - test_size
    train_data = shuffle_data.iloc[:train_size]
    valid_data = shuffle_data.iloc[train_size:train_size+valid_size]
    test_data = shuffle_data.iloc[train_size+valid_size:]
    return train_data, valid_data, test_data


raw_data = pd.read_csv('/home/yanrui/ICD/data/ICD_raw_data.csv', names=entry_names + label_name, header=None)
dropna_data = raw_data.dropna(axis=0, how='all', subset=entry_names).reset_index(drop=True)
filter_data = data_filter(dropna_data,filter_num)
train_data, valid_data, test_data = data_split(filter_data,valid_rate,test_rate)
train_data.to_csv('/home/yanrui/ICD/data/train_data.csv',index=False, header=None)
valid_data.to_csv('/home/yanrui/ICD/data/valid_data.csv',index=False, header=None)
test_data.to_csv('/home/yanrui/ICD/data/test_data.csv',index=False, header=None)

