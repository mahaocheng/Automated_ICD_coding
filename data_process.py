import pandas as pd
import numpy as np
import re
from itertools import chain
from collections import Counter
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


class DataProcess(object):
    def __init__(self, add_entry_emb, entry_names, label_name, stop_words):
        self.add_entry_emb = add_entry_emb
        self.entry_names = entry_names
        self.label_name = label_name
        self.stop_words = stop_words

    def load_data(self,data_file):
        data = pd.read_csv(data_file, names=self.entry_names + self.label_name, header=None)
        text = data[self.entry_names]
        label = list(data[''.join(self.label_name)])
        return text, label

    def clean_str(self, text_list):
        clean_sentences = []
        for sentence in text_list:
            sentence = str(sentence).replace('nan','').lower()    # 去掉文本中因缺失值含有的'nan'符号
            sentence = re.sub('[／\d\-，。：“”℃%、,+？；/（）()\.\r\n:×^‘’\']',' ', sentence)  # 去掉标点符号
            # 去停用词
            for word in self.stop_words:
                sentence = sentence.replace(word,' ')
            clean_sentences.append(sentence)
        assert len(clean_sentences) == len(text_list), 'text is None after clean!'
        return clean_sentences


    # 合并各个条目的文本
    def merge_entry_text(self, text):
        merge_text = text[self.entry_names[0]]
        text['blank'] = [' ']*len(text)
        for i in range(1,len(self.entry_names)):
            merge_text += text['blank'].map(str)  # 各条目文本之间用空格隔开，防止最后一个词与第一个词合并成一个词
            merge_text += text[self.entry_names[i]].map(str)
        merge_text = self.clean_str(merge_text)
        return merge_text


    # 建立词语-索引字典和条目-索引字典
    def build_vocab(self, merge_text, word_freq):
        split_sentences = [sentence.split() for sentence in merge_text]  # 切词
        words_list = list(chain(*split_sentences))  # 获取全部词语列表
        words_freq_dict = dict(Counter(words_list))  # 统计词频
        keep_words = []
        # 去掉低频词（词频小于word_freq的词）
        for word, freq in words_freq_dict.items():
            if freq >= word_freq:
                keep_words.append(word)
        word_vocab = {}
        entry_vocab = {}
        for i, word in enumerate(keep_words, 1):
            word_vocab[word] = i
        for i, name in enumerate(self.entry_names, 1):
            entry_vocab[name] = i
        return word_vocab, entry_vocab


    def text_to_index(self, text, word_vocab, entry_vocab):
        # 将所有条目下的文本转为index的形式
        all_text_index = []
        for name in self.entry_names:
            entry_id = entry_vocab[name]
            entry_text = self.clean_str(list(text[name]))
            entry_text_index = []
            for sentence in entry_text:
                words = sentence.split()
                if self.add_entry_emb:
                    sentence_to_index = [[entry_id, word_vocab[word]] for word in words if word in word_vocab]
                else:
                    sentence_to_index = [word_vocab[word] for word in words if word in word_vocab]
                # 如果文本中所有的词都不在词典里，补0
                if sentence_to_index == []:
                    if self.add_entry_emb:
                        sentence_to_index = [[0,0]]
                    else:
                        sentence_to_index = [0]
                entry_text_index.append(sentence_to_index)
            assert len(entry_text_index) == len(text), 'the size of entry text is not equal to sample size!'
            all_text_index.append(entry_text_index)
        # 合并所有的index形式的条目文本
        merge_index_text = []
        for i in range(len(text)):
            sentence = []
            for entry_text in all_text_index:
                sentence += entry_text[i]
            merge_index_text.append(sentence)
        return merge_index_text

    # 将label转为one-hot形式
    def label_to_onehot(self, label):
        one_hot_label = pd.get_dummies(label)
        label_list = one_hot_label.columns.values.tolist()
        return np.array(one_hot_label), label_list

    # 基于tfidf算法抽取各个ICD编码的关键词
    def extract_keywords(self, text, label, keyword_num):
        merge_text = self.merge_entry_text(text)
        # 将同一类的文本合并，构建每类ICD编码的语料库
        label_text = {}
        for i in range(len(label)):
            if label[i] not in label_text:
                label_text[label[i]] = [merge_text[i]]
            else:
                label_text[label[i]].append(merge_text[i])
        label_list = list(label_text.keys())
        corpus = []
        for key, value in label_text.items():
            corpus.append(' '.join(value))
        vectorizer = CountVectorizer()
        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
        words = vectorizer.get_feature_names()   # tfidf的词典
        weight = tfidf.toarray()   # 词典的tfidf矩阵：行代表类，列代表词
        total_keywords = []   # 关键词集
        label_keywords = {}   # 各个ICD编码对应的关键词
        for i in range(len(weight)):
            keywords = []
            word_tfidf_dict = dict(zip(words, weight[i]))   # 每一类ICD编码的tfidf分布字典
            sorted_dict = sorted(word_tfidf_dict.items(), key=lambda x: x[1], reverse=True)  # 从大到小排序
            # 选取tfidf值最大的前keyword_num个词作为关键词
            for item in sorted_dict[:keyword_num]:
                keywords.append(item[0])
            total_keywords.append(keywords)
            label_keywords[label_list[i]] = ' '.join(keywords)
        total_keywords = list(chain(*total_keywords))
        return total_keywords, label_keywords
