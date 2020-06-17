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
from keras.layers import LSTM
import pandas as pd
import numpy as np


# *******************************************************************************
# 采用LSTM模型时，第一步需要对数据进行适配处理，其中包括将数据集转化为有监督学习问题和归一化变量（包括输入和输出值），
# 使其能够实现通过前一个时刻（t-1）的污染数据和天气条件预测当前时刻（t）的污染。
# LSTM数据准备，将数据集转化为有监督学习问题和归一化变量
# 下面代码中首先加载“pollution.csv”文件，并利用sklearn的预处理模块对类别特征“风向”进行编码，当然也可以对该特征进行one-hot编码。
# 接着对所有的特征进行归一化处理，然后将数据集转化为有监督学习问题，同时将需要预测的当前时刻（t）的天气条件特征移除

# convert series to supervised learning
# 将series转化为监督学习
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):   # data是个二维表格
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
        if values[i, 0] != values[i, -9]:
            agg.drop(i, inplace=True)
    return agg


# 加载数据
dataset = read_csv('C:/Users/xue/Desktop/毕设/数据/data_txt/beijing_before_all.csv', header=0, index_col=0)  # 返回值为DataFrame类型
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
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
# 去除不需要预测的列，此处需要预测的列只有AQI
reframed.drop(reframed.columns[[9, 11, 12, 13, 14, 15, 16, 17]], axis=1, inplace=True)
print(reframed.head())

# *******************************************************************************
# 构造LSTM模型
# 将处理后的数据集划分为训练集和测试集，仅利用第一年数据进行训练，然后利用剩下的4年进行评估
# 下面的代码将数据集进行划分，然后将训练集和测试集划分为输入和输出变量，
# 最终将输入（X）改造为LSTM的输入格式，即[samples,timesteps,features]。

# split into train and test sets  分为训练集和测试集
n_train_hours = 14000
train = values[:n_train_hours, :]   # 前n_train_hours行
test = values[n_train_hours:, :]   # 后n_train_hours行
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]   # 训练集的输入和输出变量
test_X, test_y = test[:, :-1], test[:, -1]   # 测试集的输入和输出变量
# reshape input to be 3D [samples, timesteps, features]
# 将输入X改造为LSTM的输入格式，即[samples,timesteps,features]。
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
# (8760, 1, 8) (8760,) (35039, 1, 8) (35039,)

# 现在可以搭建LSTM模型了。
# LSTM模型中，隐藏层有50个神经元，输出层1个神经元（回归问题），
# 输入变量是一个时间步（t-1）的特征，损失函数采用Mean Absolute Error(MAE)，优化算法采用Adam，
# 模型采用50个epochs并且每个batch的大小为72。
# 最后，在fit()函数中设置validation_data参数，记录训练集和测试集的损失，并在完成训练和测试后绘制损失图。

# design network
model = Sequential()
model.add(LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2,
                    shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# 接下来我们对模型效果进行评估。
# 值得注意的是：需要将预测结果和部分测试集数据组合然后进行比例反转（invert the scaling），
# 同时也需要将测试集上的预期值也进行比例转换。
# 通过以上处理之后，再结合RMSE（均方根误差）计算损失。

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
t1 = concatenate((test_X[:, 0].reshape(len(test_X[:, 0]), 1), yhat), axis=1)
inv_yhat = concatenate((t1, test_X[:, 2:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 1]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
t2 = concatenate((test_X[:, 0].reshape(len(test_X[:, 0]), 1), test_y), axis=1)
inv_y = concatenate((t2, test_X[:, 2:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 1]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print(inv_y[:100])
print(inv_yhat[:100])
print('Test RMSE: %.3f' % rmse)



