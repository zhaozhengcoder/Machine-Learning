import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import mean_squared_error

"""
from keras.models import Sequential 
from keras.layers import Dense 
from keras.layers import LSTM 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import mean_squared_error
"""

import numpy as np
import pickle
import tensorflow as tf
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt


# 原始的data里面的数据格式是dataframe，arr改成了里面也是list
def transfer(data):
    vol_col_index = 1  # 找到流量对应的列
    height = len(data)
    width = data[0].shape[0]
    arr = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            arr[i, j] = data[i].iloc[j, vol_col_index]
    return arr


def createdataset(data):
    dataset = []
    for road in data:  # 对于某一条路的数据
        dataset.append(transfer(road))
    return dataset


def myload():
    filename = 'dump2.txt'
    #f = open('./data/' + filename, 'rb')
    abspath='C:/Users/wwwa8/Documents/GitHub/Machine-Learning/序列预测/PCA去趋势化/dev2/data/'
    f = open(abspath + filename, 'rb')
    data = pickle.load(f)
    f.close()
    # print (data)   # 路段数 * 每个路段的信息（df的数据结构）
    return data


def split_dataset(dataset):
    trainX = []
    trainY = []
    trainX_len = 2  # 使用3天预测一天
    trainY_len = 1  # 使用3天预测一天
    days = dataset[0].shape[0]
    for road in dataset:
        trainX_per_road = []
        trainY_pre_road = []
        for i in range(0, days - (trainX_len + trainY_len - 1)):
            trainX_per_road.append(road[i:i + trainX_len])
            trainY_pre_road.append(road[i + trainX_len:i + trainX_len + trainY_len])
        trainX.append(trainX_per_road)
        trainY.append(trainY_pre_road)
    return np.array(trainX), np.array(trainY)


def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

def get_metrics(y, pred_y):
    #替换为0的元素
    y_mean=np.mean(y)
    y[y==0.00] =y_mean
    mre = np.mean(np.abs(y - pred_y) / y)
    mae = np.mean(np.abs(y - pred_y))
    rmse = np.sqrt(np.mean(np.square(y - pred_y)))
    return mre, mae, rmse



#从文件加载数据
def load_data():
    path='123.csv'
    # load the dataset
    dataframe = read_csv(path, usecols=[0], engine='python', skipfooter=0)
    dataset = dataframe.values
    # 将整型变为float
    dataset = dataset.astype('float32')
    ## show
    #print (dataset)
    print ("shape : ",dataset.shape)
    #plt.plot(dataset,'.')
    #plt.show()
    return dataset


def split_data(dataset):
    # split into train and test sets
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    return train,test

"""
create_dataset 的作用是，划分数据 
原始的dataset = [100,200,300,400,500,600,700,800]
dataX = 
[[ 100.  200.  300.]
 [ 200.  300.  400.]
 [ 300.  400.  500.]
 [ 400.  500.  600.]
 [ 500.  600.  700.]
 [ 600.  700.  800.]]
dataY = 
[ 400.  500.  600.  700.  800.  900.]
"""
def create_dataset(dataset,look_back=3):
    dataX, dataY = [], [] 
    for i in range(len(dataset)-look_back-1): 
        a = dataset[i:(i+look_back), 0] 
        dataX.append(a) 
        dataY.append(dataset[i + look_back, 0]) 
    return np.array(dataX), np.array(dataY)



if __name__=="__main__":
    time_step = 2
    data = myload()  # data是一个list，里面是df格式的数据
    # transfer
    dataset = createdataset(data)                  # dataset 的格式是 （路段 * 每一天 * 一天内的数据）20 * 14 * 480

    #dataset_main, dataset_rest = use_pca(dataset)  # dataset_main = 路段 * 每一个路段里面的主成分 ; dataset_rest = 路段 * 每一个路段里面的偏差
    road_num = len(dataset)    
    days, dnum =dataset[0].shape #days =14,dnum=480

    #y_main = dataset_main[0][time_step:days, :]    # y_main的主成分[2~14] shape 12 * 2 * 480
    #将偏差数据 分割 成输入和输出
    trainX, trainY = split_dataset(dataset)   #trainX 的shape 20 * 12 * 2 *480 ； trainY 的shape 20 * 12 * 1 *480

    train_x_raw = trainX[0]                                       # 取了第一条路段来进行预测 train_x_raw的sahpe ：12 * 2 *480
    train_y_raw = np.reshape(trainY[0], (days - time_step, 480))  # train_y_raw的shape 12 * 480
    train_x_raw = np.reshape(train_x_raw, (days - time_step,time_step*dnum))   #train_x_raw (12, 2*480)
    train_y_raw = np.reshape(train_y_raw, (days - time_step,dnum))             #train_y_raw (12,  480)

    # 归一化
    x_max = train_x_raw.max()
    x_min = train_x_raw.min()
    y_max = train_y_raw.max()
    y_min = train_y_raw.min()

    train_x = (train_x_raw - x_min) / (x_max - x_min)
    train_y = (train_y_raw - y_min) / (y_max - y_min)

    # 构造训练和测试集
    test_split_rate=0.25
    total_len = train_x.shape[0]
    train_len = int(total_len * (1-test_split_rate))
    test_len = int(total_len * test_split_rate)

    test_x = train_x[train_len:]                # (3,2*480)
    test_y = train_y[train_len:]                # (3, 480)
    #train_x =train_x[0:train_len]
    #train_y =train_y[0:train_len]
    train_x =train_x                            # (12,2*480)
    train_y =train_y                            # (12, 480)

    #train_x = np.reshape(train_x,(12*480,2))
    #train_y = np.reshape(train_y,(12*480,1))
    
    
    test_x = np.reshape(test_x,(3*480,2))
    test_y = np.reshape(test_y,(3*480,1))
    train_x = test_x
    train_y = test_y
    """
    train_x.shape :  (1261, 3)
    train_y.shape :  (1261, 1)  
    test_x.shape :  (620, 3)
    test_y.shape :  (620, 1)
    """

    #todo svr method 
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.01)
    #svr_lin = SVR(kernel='linear', C=1e3)
    #svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    y_rbf = svr_rbf.fit(train_x,train_y).predict(test_x)
    #y_lin = svr_lin.fit(train_x, train_y).predict(test_x)
    #y_poly = svr_poly.fit(train_x, train_y).predict(test_x)


    #反标准化
    #y_rbf=np.reshape(y_rbf,(len(y_rbf),-1))
    #y_rbf=scaler.inverse_transform(y_rbf)
    #test_y=scaler.inverse_transform(test_y)
    
    plt.plot(y_rbf,label='y_rbf')
    #plt.plot(y_lin,label='y_lin')
    #plt.plot(y_poly,label='y_poly')
    plt.plot(test_y,label='true')
    plt.legend(loc='upper right')
    plt.show()

    #print (dataset)

    #rmse
    testScore = math.sqrt(mean_squared_error(y_rbf,test_y))
    print('Test Score: %.2f RMSE' % (testScore))
    ## Test Score: 11.58 RMSE
