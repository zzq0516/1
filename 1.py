import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn import preprocessing
import math
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt



np.set_printoptions(threshold=np.inf)

# 全局参数
Splid_ratio = 0.6 # 训练集比例
Look_back = 10   # 回顾系数
epochs = 10000  # 迭代轮次

piao_fang = pd.read_csv('./data/1.csv', index_col=None, delimiter=',')  # 打开数据集
labels = piao_fang['money'].values  # 提取值
dataset = labels.reshape(-1, 1)  # 改变数组形状


Scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))  # 归一化处理
dataset = Scaler.fit_transform(dataset)

# 分块
split_size = int(len(dataset) * Splid_ratio)
test_size = len(dataset) - split_size

train_dataset = dataset[0:split_size, :]
test_dataset = dataset[split_size:len(dataset), :]
a = Scaler.fit_transform(test_dataset)



def buildDataset(dataset, Look_back):  # 分离处理出训练值
    dataX = []
    dataY = []
    for i in range(len(dataset) - Look_back - 1):
        a = dataset[i:(i + Look_back), 0]
        dataX .append(a)
        dataY.append(dataset[(i + Look_back), 0])
    return np.array(dataX), np.array(dataY)


train_x, train_y = buildDataset(train_dataset, Look_back)
test_x, test_y = buildDataset(test_dataset, Look_back)


train_x = np.reshape(train_x,(train_x.shape[0],1,train_x.shape[1]))  # 对数组进行重置使其符合模型输入要求
test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))
print(test_x)


# 模型的建立
Model = Sequential()
Model.add(LSTM(250, input_shape=(1, Look_back)))
Model.add(Dropout(0.1))
Model.add(Dense(1))
Model.compile(loss='mse', optimizer='adam')  # 使用adam优化器
Model.fit(train_x, train_y, epochs=epochs, batch_size=Look_back, verbose=0)  # 开始训练

train_predict = Model.predict(train_x)  # 预测
test_predict = Model.predict(test_x)
print(test_predict)

# 对预测数据的处理
train_predict = Scaler.inverse_transform(train_predict)
train_y = Scaler.inverse_transform([train_y])
test_predict = Scaler.inverse_transform(test_predict)
test_y = Scaler.inverse_transform([test_y])


# 计算误差分数
train_score = math.sqrt(mean_squared_error(train_y[0], train_predict[:, 0]))
print(f'训练误差分数：{np.round(train_score,3)}')
test_score = math.sqrt(mean_squared_error(test_y[0], test_predict[:, 0]))
print(f'测试误差分数：{np.round(test_score,3)}')

# 对数据进行可视化处理
train_predict_plot = np.empty_like(dataset)
train_predict_plot[:, :] = np.nan
train_predict_plot[Look_back:len(train_predict) + Look_back, :] = train_predict

test_predict_plot = np.empty_like(dataset)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict) + Look_back * 2 +
                  1:len(dataset) - 1, :] = test_predict

# 绘图
plt.figure(figsize=(20, 7))
plt.plot(Scaler.inverse_transform(dataset))

plt.plot(train_predict_plot)
plt.plot(test_predict_plot)
plt.show()
