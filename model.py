import tensorflow as tf
import numpy as np
import datetime
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

class ICDmodel(object):
    """
        Our proposed model for automated ICD coding.
        The framework is a two-layer BiLSTM text encoder and a full connected network classifier.
        Text encoder includes three modules:
        1. word embedding with entry embedding module;
        2. keyword attention module;
        3. word attention module.
    """
    def __init__(self, vocab_size, num_classes, num_entries, word_emb_dim, entry_emb_dim, hidden_dim,
                 learning_rate, keep_prob, add_entry_emb, add_keyword_attention, add_word_attention, keywords_id):
        self.add_entry_emb = add_entry_emb
        # 输入层
        with tf.name_scope('input_layer'):
            if add_entry_emb:
                self.input_x = tf.placeholder(tf.int32, [None, None, 2])
            else:
                self.input_x = tf.placeholder(tf.int32, [None, None])
            self.input_y = tf.placeholder(tf.float32, [None, num_classes])
            self.input_seqlen = tf.placeholder(tf.int32, [None])

        # word embedding 层
        with tf.variable_scope('embedding_layer'):
            word_embed = tf.get_variable('word_embed', shape=[vocab_size + 1, word_emb_dim], dtype=tf.float32)
            entry_embed = tf.get_variable('entry_embed', shape=[num_entries + 1, entry_emb_dim], dtype=tf.float32)
            if add_entry_emb:
                input_entry_emb = tf.nn.embedding_lookup(entry_embed, self.input_x[:, :, 0])
                input_word_emb = tf.nn.embedding_lookup(word_embed, self.input_x[:, :, 1])
                input_emb = tf.concat([input_entry_emb, input_word_emb], 2)
            else:
                input_emb = tf.nn.embedding_lookup(word_embed, self.input_x)

        # 第一层双向LSTM
        with tf.variable_scope('first_BiLSTM_layer'):
            fw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_dim, state_is_tuple=True)
            bw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_dim, state_is_tuple=True)
            fw_dropoutcell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=keep_prob)
            bw_dropoutcell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=keep_prob)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_dropoutcell, bw_dropoutcell, inputs=input_emb,
                                                         sequence_length=self.input_seqlen, dtype=tf.float32)
            first_outputs = tf.concat(outputs, 2)
            print(first_outputs.shape)

        # keyword attention模块
        if add_keyword_attention:
            keyword_emb = tf.nn.embedding_lookup(word_embed, keywords_id)
            with tf.variable_scope('keyword_attention'):
                keyword_atten_w = tf.get_variable('keyword_atten_w', [word_emb_dim, hidden_dim * 2])
                keyword_atten_b = tf.get_variable('keyword_atten_b', [hidden_dim * 2])
                keyword_representation = tf.tanh(tf.matmul(keyword_emb, keyword_atten_w) + keyword_atten_b)
                print(keyword_representation.shape)  # (100, 100)
                atten_dot = tf.tensordot(first_outputs, tf.transpose(keyword_representation), axes=1)
                alphas = tf.nn.softmax(atten_dot)
                print(alphas.shape)  # (?, 500,100)
                atten_outputs = tf.reduce_sum(tf.expand_dims(alphas, -1) * keyword_representation, 2)
                print('first attention output shape:', atten_outputs.shape)  # (?, 500, 200)
                second_inputs = tf.concat([first_outputs, atten_outputs], 2)
        else:
            second_inputs = first_outputs
        print('sencond_outputs shape: ', second_inputs.shape)

        # 第二层双向LSTM
        with tf.variable_scope('second_BiLSTM_layer'):
            fw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_dim, state_is_tuple=True)
            bw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_dim, state_is_tuple=True)
            fw_dropoutcell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=keep_prob)
            bw_dropoutcell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=keep_prob)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_dropoutcell, bw_dropoutcell, inputs=second_inputs,
                                                         sequence_length=self.input_seqlen, dtype=tf.float32)
            second_outputs = tf.concat(outputs, 2)

        # word attention模块
        if add_word_attention:
            with tf.variable_scope('word_attention'):
                word_atten_v = tf.get_variable('word_atten_v', [hidden_dim * 2], dtype=tf.float32)
                atten_dot = tf.tensordot(second_outputs, tf.transpose(word_atten_v), axes=1)
                alphas = tf.nn.softmax(atten_dot)
                print('alphas shape: ', alphas.shape)
                last_outputs = tf.reduce_sum(tf.expand_dims(alphas, -1) * second_outputs, 1)
        else:
            last_outputs = tf.reduce_mean(second_outputs, axis=1)
        print('last_outputs shape: ', last_outputs.shape)

        # 全连接层，这里只用了一层，可增加一层
        with tf.variable_scope('full_connect_layer'):
            logit = tf.layers.dense(last_outputs, num_classes)
            self.y_pred = tf.argmax(tf.nn.softmax(logit), 1)

        # 优化过程
        with tf.variable_scope('optimize'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            self.optimize = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

        # 计算训练过程中每个batch的准确率
        with tf.variable_scope('accuracy'):
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred)
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # 补齐每个batch的长度
    def batch_seq_padding(self, sequences, max_length):
        seq_padding = []
        seq_len = [len(seq) for seq in sequences]
        max_seqlen = max(seq_len)
        # 是否对序列进行截断，默认不截断
        if max_length != None:
            if max_seqlen > max_length:
                max_seqlen = max_length
        for seq in sequences:
            if len(seq) < max_seqlen:
                if self.add_entry_emb:
                    seq = seq + [[0, 0]] * (max_seqlen - len(seq))
                else:
                    seq = seq + [0] * (max_seqlen - len(seq))
            else:
                seq = seq[:max_seqlen]
            # 验证每个batch的输入序列长度一致
            assert len(seq) == max_seqlen, 'the sequences length of input batch is not inconsistent!'
            seq_padding.append(seq)
        return seq_padding, seq_len

    # 训练过程
    def training(self, train_x, train_y, test_x, test_y, batch_size, num_epochs, label_list, max_length):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(num_epochs):
                print('training epoch {}'.format(epoch+1))
                for i in range(0, len(train_x), batch_size):
                    j = i + batch_size
                    assert train_x[i:j] != [], 'empty input!'
                    batch_x, batch_seqlen = self.batch_seq_padding(train_x[i:j], max_length)
                    batch_y = train_y[i:j]
                    feed_dict = {self.input_x: batch_x, self.input_y: batch_y, self.input_seqlen: batch_seqlen}
                    train_loss, train_acc, _ = sess.run([self.loss, self.accuracy, self.optimize], feed_dict=feed_dict)
                    batch_id = int(i / batch_size)
                    if batch_id % 20 == 0:
                        time_str = datetime.datetime.now().isoformat()
                        print('{}：batch {}, train_loss = {}, train_acc = {}'.format(time_str, batch_id, train_loss,train_acc))
                print('epoch {} training done!'.format(epoch+1))
                print('Evaluate ...')
                test_y_true = list(np.argmax(test_y, 1))
                test_y_pred = []
                for m in range(0, len(test_x), batch_size):
                    n = m + batch_size
                    assert test_x[m:n] != [], 'empty input!'
                    batch_test_x, batch_test_seqlen = self.batch_seq_padding(test_x[m:n], max_length)
                    batch_test_y = test_y[m:n]
                    feed_dict = {self.input_x: batch_test_x, self.input_y: batch_test_y, self.input_seqlen: batch_test_seqlen}
                    test_pred = sess.run(self.y_pred, feed_dict=feed_dict)
                    test_y_pred += list(test_pred)

                assert len(test_y_true) == len(test_y_pred), 'y_pred and y_true is not equal length!'

                accuracy = accuracy_score(test_y_true, test_y_pred)
                precision = precision_score(test_y_true, test_y_pred, average='macro')
                recall = recall_score(test_y_true, test_y_pred, average='macro')
                f1 = f1_score(test_y_true, test_y_pred, average='macro')
                print('accuracy = {}, precision = {}'.format(accuracy, precision))
                print('recall = {}, f1_score = {}'.format(recall, f1))
                print('===============================================================================================')

            final_results_report = classification_report(test_y_true, test_y_pred, target_names=label_list, digits=4)
            print(final_results_report)
            # 保存结果
            with open('./results/results.txt','w') as f:
                line = 'accuracy = {}, precision = {}, recall = {}, f1_score = {}'.format(accuracy, precision,recall, f1)
                f.write(line + '\n')
                f.write(final_results_report)
            print('done!')
