import numpy as np
from data_process import DataProcess
from model import ICDmodel
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# 设置超参数
keyword_num = 10                   # 关键词个数
word_freq = 5                      # 最低词频
hidden_dim = 128                   # LSTM的隐层神经元个数（输出维度）
word_emb_dim = 128                 # 词向量维度
entry_emb_dim = 32                 # 条目向量维度
max_length = None                  # 截断的最大长度，默认None，为不截断，可根据序列的平均长度和最大长度进行调整
keep_prob = 0.8                    # dropout保留比例
batch_size = 100                   # 每个batch的大小
learning_rate = 0.01               # 学习率
num_epochs = 10                    # 数据重复训练次数
add_entry_emb = True               # 是否加条目词嵌入
add_keyword_attention = True       # 是否加关键词注意力
add_word_attention = True          # 是否加词语注意力


entry_names = ['主诉', '现病史', '检查报告', '首次病程记录', '查房记录', '出院记录']
label_name = ['ICD编码']
# 自定义停用词，可酌情增加
stop_words = ['患者', '正常', '明显', '每次', '入院', '出院', '年', '月', '日', '天', '术后', '考虑', '显示', '我院', '外院']

print('loading data ...')
process = DataProcess(add_entry_emb, entry_names, label_name, stop_words)
train_text, train_label = process.load_data('../data/train_data.csv')
test_text, test_label = process.load_data('../data/test_data.csv')

print('building word and entry vocabulary...')
merge_text = process.merge_entry_text(train_text)
word_vocab, entry_vocab = process.build_vocab(merge_text, word_freq)
vocab_size = len(word_vocab)
num_entries = len(entry_names)

print('transform text to index ...')
train_x = process.text_to_index(train_text, word_vocab, entry_vocab)
test_x = process.text_to_index(test_text, word_vocab, entry_vocab)

num_classes = len(set(train_label))
test_classes = len(set(test_label))
train_y, label_list = process.label_to_onehot(train_label)
test_y, _ = process.label_to_onehot(test_label)


sentences_length = [len(x) for x in train_x]
mean_seq_length = np.mean(sentences_length)
max_seq_length = np.max(sentences_length)

print('--------------------------------------------------')
print('data set information:')
print('word_vocab_size = {}, entry_vocab_size = {}'.format(vocab_size, num_entries))
print('train size = {}, test size = {}'.format(len(train_x),len(test_x)))
print('mean_seq_length = {}, max_seq_length = {}'.format(mean_seq_length, max_seq_length))
print('train_num_classes = {}, test_num_classes = {}'.format(num_classes, test_classes))

if add_keyword_attention:
    print('--------------------------------------------------')
    print('extract key words ...')
    total_keywords, label_keywords = process.extract_keywords(train_text, train_label, keyword_num)
    for key, value in label_keywords.items():
        print(key, ':', value)
    keywords_id = [word_vocab[word] for word in total_keywords if word in word_vocab]
    print('--------------------------------------------------')
else:
    keywords_id = None

print('loading model ...')
model = ICDmodel(vocab_size, num_classes, num_entries, word_emb_dim, entry_emb_dim, hidden_dim, learning_rate,
                keep_prob, add_entry_emb, add_keyword_attention, add_word_attention, keywords_id)
print('start training and testing model ...')
model.training(train_x, train_y, test_x, test_y, batch_size, num_epochs, label_list, max_length)
