from __future__ import print_function, division

import numpy as np
import tensorflow as tf
from keras.datasets import imdb
from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from tqdm import tqdm
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import concatenate

from attention import attention
from utils import get_vocabulary_size, fit_in_vocabulary, zero_pad, batch_generator

# 双向GRU+Attention模型

HIDDEN_SIZE = 64
ATTENTION_SIZE = 16
KEEP_PROB = 1
BATCH_SIZE = 256
NUM_EPOCHS = 50   # 迭代次数
DELTA = 0.5   # 对于训练模型没有用，只是用来计算总迭代次数的损失
MODEL_PATH = './model'
n_hours = 100
n_features = 9


# *******************************************************************************
# 采用LSTM模型时，第一步需要对数据进行适配处理，其中包括将数据集转化为有监督学习问题和归一化变量（包括输入和输出值），
# 使其能够实现通过前一个时刻（t-1）的污染数据和天气条件预测当前时刻（t）的污染。
# LSTM数据准备，将数据集转化为有监督学习问题和归一化变量
# 下面代码中首先加载“pollution.csv”文件，并利用sklearn的预处理模块对类别特征“风向”进行编码，当然也可以对该特征进行one-hot编码。
# 接着对所有的特征进行归一化处理，然后将数据集转化为有监督学习问题，同时将需要预测的当前时刻（t）的天气条件特征移除

# convert series to supervised learning
# 将series转化为监督学习
def series_to_supervised(data, n_features, n_in=1, n_out=1, dropnan=True):  # data是个二维表格
    n_vars = 1 if type(data) is list else data.shape[1]  # .shape获取维度，这里的shape[1]是取了列数
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)  输入序列
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))  # shift默认是从上往下移
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)  预测序列
    for i in range(0, n_out):
        cols.append(df.shift(-i))  # 这里是从下往上移，i为0时不移动
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together  横着连接
    agg = concat(cols, axis=1)  # cols是list类型，每个元素是DataFrame类型的数据
    agg.columns = names  # 指定列名
    # drop rows with NaN values   删除有NaN值的行
    if dropnan:
        agg.dropna(inplace=True)
    # 删除地点不一样的行
    values = agg.values
    for i in range(agg.shape[0]):
        if (values[i, 0] != values[i, -n_features]) or (values[i, -2 * n_features] != values[i, -n_features]):
            agg.drop(i, inplace=True)
    return agg


# 加载数据
dataset = read_csv('D:/PycharmSpace/fog/data/beijing_before_all.csv', header=0, index_col=0)  # 返回值为DataFrame类型
dataset.drop(['time', 'aqi_type', 'first_pollution'], axis=1, inplace=True)  # 删除不需要的列
values = dataset.values
# integer encode direction  对监测点名称进行数字化的编码
encoder = LabelEncoder()
values[:, 0] = encoder.fit_transform(values[:, 0])
# 确保所有数据都是float类型
values = values.astype('float32')
# normalize features  对特征进行归一化
scaler = MinMaxScaler(feature_range=(0, 1))  # 进行归一化，范围[0,1]
scaled = scaler.fit_transform(values)  # scaled是个二维表格

# frame as supervised learning
reframed = series_to_supervised(scaled, n_features, n_hours, 1)
# 去除不需要预测的列，此处需要预测的列只有AQI
print(reframed.head())


# split into train and test sets  分为训练集和测试集
values = reframed.values
length = len(values)
n_train_hours = int(length * 0.8)
train = values[:n_train_hours, :]  # 前n_train_hours行
test = values[n_train_hours:, :]

# split into input and outputs
n_x = n_hours * n_features  # X的列数
n_y = -(n_features - 1)  # Y的列数
X_train, y_train = train[:, :n_x], train[:, n_y]  # 训练集的输入和输出变量
X_test, y_test = test[:, :n_x], test[:, n_y]  # 测试集的输入和输出变量

# Different placeholders
with tf.name_scope('Inputs'):
    batch_ph = tf.placeholder(tf.float32, [None, n_hours, n_features], name='batch_ph')  # & 长度为每个时刻的特征数*时刻数
    target_ph = tf.placeholder(tf.float32, [None], name='target_ph')
    # seq_len_ph = tf.placeholder(tf.int32, [None], name='seq_len_ph')
    keep_prob_ph = tf.placeholder(tf.float32, name='keep_prob_ph')

train_size = X_train.shape[0]

batch_input = batch_ph
rnn_outputs, _ = bi_rnn(GRUCell(HIDDEN_SIZE), GRUCell(HIDDEN_SIZE),
                        inputs=batch_input, dtype=tf.float32)

# 注意力层
with tf.name_scope('Attention_layer'):
    attention_output, alphas = attention(rnn_outputs, ATTENTION_SIZE, return_alphas=True)

# Dropout
drop = tf.nn.dropout(attention_output, keep_prob_ph)

# 全连接层
with tf.name_scope('Fully_connected_layer'):
    W = tf.Variable(tf.truncated_normal([HIDDEN_SIZE * 2, 1], stddev=0.1))  # Hidden size is multiplied by 2 for Bi-RNN
    b = tf.Variable(tf.constant(0., shape=[1]))
    y_hat = tf.nn.xw_plus_b(drop, W, b)
    y_hat = tf.squeeze(y_hat)  # 预测结果

with tf.name_scope('Metrics'):
    # 以MAE为损失函数
    loss = tf.convert_to_tensor(tf.reduce_mean(tf.abs(target_ph - y_hat)), dtype=tf.float32)
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

    # Accuracy metric 以RMSE作为度量指标
    accuracy = tf.convert_to_tensor(tf.sqrt(tf.reduce_mean(tf.square(target_ph - y_hat))), dtype=tf.float32)
    tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()

# Batch generators  生成每个batch的数据
train_batch_generator = batch_generator(X_train, y_train, BATCH_SIZE)
test_batch_generator = batch_generator(X_test, y_test, BATCH_SIZE)

train_writer = tf.summary.FileWriter('./logdir/train', accuracy.graph)
test_writer = tf.summary.FileWriter('./logdir/test', accuracy.graph)

session_conf = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

saver = tf.train.Saver()

if __name__ == "__main__":
    train_label = np.zeros(shape=(10,))
    train_hat = np.zeros(shape=(10,))
    test_sample = np.zeros(shape=(10,))
    test_hat = np.zeros(shape=(10,))
    test_label = np.zeros(shape=(10,))

    with tf.Session(config=session_conf) as sess:
        sess.run(tf.global_variables_initializer())
        print("Start learning...")
        # num = 0
        for epoch in range(NUM_EPOCHS):
            loss_train = 0  # 训练集损失
            loss_test = 0  # 测试集损失
            accuracy_train = 0  # 训练集精确度
            accuracy_test = 0  # 测试集精确度
            # num += 1  # 记录迭代次数
            # 记录预测值,每次迭代清空
            i = 0
            j = 0
            train_label = np.zeros(shape=(10,))
            train_hat = np.zeros(shape=(10,))
            test_sample = np.zeros(shape=(10,))
            test_hat = np.zeros(shape=(10,))
            test_label = np.zeros(shape=(10,))
            print("epoch: {}\t".format(epoch), end="")

            # Training
            num_batches = X_train.shape[0] // BATCH_SIZE
            for b in tqdm(range(num_batches)):
                x_batch, y_batch = next(train_batch_generator)  # 取出来的一个batch的样本数据和标签
                x_batch = x_batch.reshape((x_batch.shape[0], n_hours, n_features))
                # seq_len = np.array([list(x).index(0) + 1 for x in x_batch])  # actual lengths of sequences
                loss_tr, acc, _, summary, tr_hat = sess.run([loss, accuracy, optimizer, merged, y_hat],
                                                            feed_dict={batch_ph: x_batch,
                                                                       target_ph: y_batch,
                                                                       # seq_len_ph: seq_len,
                                                                       keep_prob_ph: KEEP_PROB})
                accuracy_train += acc
                loss_train = loss_tr * DELTA + loss_train * (1 - DELTA)
                train_writer.add_summary(summary, b + num_batches * epoch)
                # 记录训练集标签和预测值
                i += 1
                if i == 1:
                    train_hat = tr_hat
                    train_label = y_batch
                elif i > 1:
                    train_hat = concatenate((train_hat, tr_hat))
                    train_label = concatenate((train_label, y_batch))

            accuracy_train /= num_batches

            # Testing
            num_batches = X_test.shape[0] // BATCH_SIZE
            for b in tqdm(range(num_batches)):
                x_batch, y_batch = next(test_batch_generator)
                x_batch = x_batch.reshape((x_batch.shape[0], n_hours, n_features))
                # seq_len = np.array([list(x).index(0) + 1 for x in x_batch])  # actual lengths of sequences
                loss_test_batch, acc, summary, te_hat = sess.run([loss, accuracy, merged, y_hat],
                                                                 feed_dict={batch_ph: x_batch,
                                                                            target_ph: y_batch,
                                                                            # seq_len_ph: seq_len,
                                                                            keep_prob_ph: 1.0})
                accuracy_test += acc
                loss_test += loss_test_batch
                test_writer.add_summary(summary, b + num_batches * epoch)
                # 记录测试集标签和预测值
                j += 1
                if j == 1:
                    test_sample = x_batch
                    test_hat = te_hat
                    test_label = y_batch
                elif j > 1:
                    test_sample = concatenate((test_sample, x_batch))
                    test_hat = concatenate((test_hat, te_hat))
                    test_label = concatenate((test_label, y_batch))
            accuracy_test /= num_batches
            loss_test /= num_batches

            print("epoch--{:.0f},loss: {:.3f}, val_loss: {:.3f}, acc: {:.3f}, val_acc: {:.3f}".format(
                epoch + 1, loss_train, loss_test, accuracy_train, accuracy_test
            ))
        train_writer.close()
        test_writer.close()
        saver.save(sess, MODEL_PATH)
        print("Run 'tensorboard --logdir=./logdir' to checkout tensorboard logs.")

    print('Finish')

# 转换比例
# invert scaling for forecast
test_sample = test_sample.reshape((test_sample.shape[0], n_hours * n_features))
t1 = concatenate((test_sample[:, 0].reshape(len(test_sample[:, 0]), 1), test_hat.reshape(len(test_hat), 1)),
                 axis=1)  # 将test_sample的第一列和test_hat连接起来
inv_yhat = concatenate((t1, test_sample[:, (-(n_features - 2)):]), axis=1)  # 将t1和test_X的最后n_features-2列连接起来
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 1]  # 取AQI列
# invert scaling for actual
test_label = test_label.reshape((len(test_label), 1))
t2 = concatenate((test_sample[:, 0].reshape(len(test_sample[:, 0]), 1), test_label), axis=1)
inv_y = concatenate((t2, test_sample[:, (-(n_features - 2)):]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 1]

# 计算误差
err = inv_yhat - inv_y
err_abs = abs(inv_yhat - inv_y)

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
# print(inv_y[:100])
# print(inv_yhat[:100])
print('Test RMSE: %.3f' % rmse)
print('Test MAE: %.3f' % float(err_abs.sum(axis=0) / len(err_abs) * 1.0))
print('Over')
