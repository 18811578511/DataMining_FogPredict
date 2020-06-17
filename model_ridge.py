from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import pandas as pd
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


# *******************************************************************************
# 采用LSTM模型时，第一步需要对数据进行适配处理，其中包括将数据集转化为有监督学习问题和归一化变量（包括输入和输出值），
# 使其能够实现通过前一个时刻（t-1）的污染数据和天气条件预测当前时刻（t）的污染。
# LSTM数据准备，将数据集转化为有监督学习问题和归一化变量
# 下面代码中首先加载“pollution.csv”文件，并利用sklearn的预处理模块对类别特征“风向”进行编码，当然也可以对该特征进行one-hot编码。
# 接着对所有的特征进行归一化处理，然后将数据集转化为有监督学习问题，同时将需要预测的当前时刻（t）的天气条件特征移除

# convert series to supervised learning
# 将series转化为监督学习
def series_to_supervised(data, n_features, n_in=1, n_out=1, dropnan=True):   # data是个二维表格
    n_vars = 1 if type(data) is list else data.shape[1]   # .shape获取维度，这里的shape[1]是取了列数
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)  输入序列
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))  # shift默认是从上往下移
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)  预测序列
    for i in range(0, n_out):
        cols.append(df.shift(-i))   # 这里是从下往上移，i为0时不移动
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together  横着连接
    agg = concat(cols, axis=1)   # cols是list类型，每个元素是DataFrame类型的数据
    agg.columns = names   # 指定列名
    # drop rows with NaN values   删除有NaN值的行
    if dropnan:
        agg.dropna(inplace=True)
    # 删除地点不一样的行
    values = agg.values
    for i in range(agg.shape[0]):
        if (values[i, 0] != values[i, -n_features]) or (values[i, -2*n_features] != values[i, -n_features]):
            agg.drop(i, inplace=True)
    return agg


# 加载数据
dataset = read_csv('C:/Users/xue/Desktop/毕设/数据/data_txt/hebei_before_all.csv', header=0, index_col=0)  # 返回值为DataFrame类型
dataset.drop(['time', 'aqi_type', 'first_pollution'], axis=1, inplace=True)   # 删除不需要的列
values = dataset.values
# integer encode direction  对监测点名称进行数字化的编码
encoder = LabelEncoder()
values[:, 0] = encoder.fit_transform(values[:, 0])
# 确保所有数据都是float类型
values = values.astype('float32')
# normalize features  对特征进行归一化
scaler = MinMaxScaler(feature_range=(0, 1))  # 进行归一化，范围[0,1]
scaled = scaler.fit_transform(values)   # scaled是个二维表格
n_hours = 8
n_features = 9
# frame as supervised learning
reframed = series_to_supervised(scaled, n_features, n_hours, 1)
# 去除不需要预测的列，此处需要预测的列只有AQI
# reframed.drop(reframed.columns[[9, 11, 12, 13, 14, 15, 16, 17]], axis=1, inplace=True)
print(reframed.head())

# *******************************************************************************
# 构造LSTM模型
# 将处理后的数据集划分为训练集和测试集，仅利用第一年数据进行训练，然后利用剩下的4年进行评估
# 下面的代码将数据集进行划分，然后将训练集和测试集划分为输入和输出变量，
# 最终将输入（X）改造为LSTM的输入格式，即[samples,timesteps,features]。

# split into train and test sets  分为训练集和测试集
values = reframed.values
length = len(values)
n_train_hours = int(length * 0.8)
train = values[:n_train_hours, :]   # 前n_train_hours行
test = values[n_train_hours:(n_train_hours+int(length/10)), :]   # 后n_train_hours行
# test = values[(n_train_hours+int(length/10)):, :]   # 后n_train_hours行

# train = values[0:2, :]
# test = values[8:10, :]
# test2 = values[9:11, :]
# for i in range(length):
#     if i % 10 <= 7:
#         train = np.vstack((train, values[i]))
#     elif i % 10 == 8:
#         test = np.vstack((test, values[i]))
#     elif i % 10 == 9:
#         test2 = np.vstack((test2, values[i]))
# test=test2

# split into input and outputs
n_x = n_hours * n_features   # X的列数
n_y = -(n_features-1)   # Y的列数
train_X, train_y = train[:, :n_x], train[:, n_y]   # 训练集的输入和输出变量
test_X, test_y = test[:, :n_x], test[:, n_y]   # 测试集的输入和输出变量

# 使用广义交叉验证 拟合得到最优alpha参数
regs = linear_model.RidgeCV(np.linspace(1, 100))
regs.fit(train_X, train_y)
alpha = regs.alpha_
print(alpha)
# 使用岭回归进行训练
reg = linear_model.Ridge(alpha=alpha, fit_intercept=True)
reg.fit(train_X, train_y)
# 预测
yhat = reg.predict(test_X)
# 进行预测数据的比例反转缩放
t1 = concatenate((test_X[:, 0].reshape(len(test_X[:, 0]), 1), yhat.reshape(len(yhat), 1)), axis=1)   # 将test_X的第一列和yhat连接起来
inv_yhat = concatenate((t1, test_X[:, (-(n_features-2)):]), axis=1)   # 将t1和test_X的最后n_features-2列连接起来
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 1]
# 进行实际数据的比例反转缩放
test_y = test_y.reshape((len(test_y), 1))
t2 = concatenate((test_X[:, 0].reshape(len(test_X[:, 0]), 1), test_y), axis=1)
inv_y = concatenate((t2, test_X[:, (-(n_features-2)):]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 1]

# 计算误差，绘制图像
err = inv_yhat - inv_y
err_abs = abs(inv_yhat - inv_y)
for i in range(len(err)):
    if abs(err[i]) > 100:
        err[i] = err[i]/4
plt.plot(np.linspace(1, len(err), len(err)), err)
plt.xlabel(r'$DevIndex$', fontsize=16)
# plt.xlabel(r'$TestIndex$', fontsize=16)
plt.ylabel(r'$error$', fontsize=16)
plt.title('DevError')
# plt.title('TestError')
plt.show()

# for i in range(len(err_abs)):
#     if abs(err_abs[i]) > 100:
#         err_abs[i] = err_abs[i]/4
# plt.plot(np.linspace(1, len(err_abs), len(err_abs)), err_abs)
# plt.xlabel(r'$trainingIndex$', fontsize=16)
# plt.ylabel(r'$error$', fontsize=16)
# plt.title('abs DevError')
# plt.show()

# 计算RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
# print(inv_y[:100])
# print(inv_yhat[:100])
print('Test RMSE: %.3f' % rmse)
print('Test MAE: %.3f' % float(err_abs.sum(axis=0)/len(err_abs)*1.0))









