import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import math

import tensorflow as tf 
from sklearn.preprocessing import MinMaxScaler 

"""
from keras.models import Sequential 
from keras.layers import Dense 
from keras.layers import LSTM 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import mean_squared_error
"""


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


def show():
    pass

if __name__=="__main__":
    dataset=load_data()

    #标准化
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    
    train_data,test_data=split_data(dataset)
    train_x,train_y=create_dataset(train_data)
    test_x ,test_y =create_dataset(test_data)

    #reshape 
    train_y=np.reshape(train_y,(len(train_y),-1))
    test_y=np.reshape(test_y,(len(test_y),-1))
    #print (train_x) 
    #print (train_x.shape)
    #print (train_y)
    #print (train_y.shape)

    #placeholder 占个位
    xs = tf.placeholder(tf.float32, [None, 3])
    ys = tf.placeholder(tf.float32, [None, 1])

    # add hidden layer
    l1 = add_layer(xs, 3, 10, activation_function=tf.nn.relu)

    # add output layer
    # 上一层的输出是这一层的输入
    prediction = add_layer(l1, 10, 1, activation_function=None)

    # the error between prediction and real data
    #loss函数和使用梯度下降的方式来求解

    #loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
    loss = tf.reduce_mean(tf.square(tf.subtract(ys,prediction)))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    if int((tf.__version__).split('.')[1]) < 12:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    for i in range(5000):
        # trainings
        sess.run(train_step, feed_dict={xs: train_x, ys: train_y})
        if i % 50 == 0:
            # to see the step improvement
            # 在带有placeholder的变量里面，每一次sess.run 都需要给一个feed_dict，这个不能省略啊！
            print("loss : ",sess.run(loss, feed_dict={xs: train_x, ys: train_y}))

            #test
            """
            pre_res=sess.run(prediction,feed_dict={xs:test_x,ys:test_y})
            print ("prediction : ",pre_res)
            print ("anti normalize : ",scaler.inverse_transform(pre_res))
            """
    pre_res=sess.run(prediction,feed_dict={xs:test_x,ys:test_y}) 
    antinormal_pre_res=scaler.inverse_transform(pre_res) 
    antinormal_test_y=scaler.inverse_transform(test_y) 
    #print ("anti normalize : ",antinormal_pre_res)
    plt.plot(antinormal_pre_res)
    plt.plot(antinormal_test_y)
    plt.show()

    #计算rmse
    rmse_loss=tf.sqrt(tf.reduce_mean(tf.squared_difference(antinormal_pre_res,antinormal_test_y)))
    print("rmse  loss : ",sess.run(rmse_loss))
    """
    print ("train_x shape : ",train_x.shape)
    print ("train_y shape : ",train_y.shape)
    print ("test_x shape : ",test_x.shape)
    print ("test_y shape : ",test_y.shape)
    print ("test_ y : ",scaler.inverse_transform(test_y))
    """
# 记录
"""
1. 换上了123.csv的数据集之后，loss在 0.00655 左右


2. 在使用1.csv的数据集的表现，每次的结果不太一样，差距还很大
loss :  1.26701e-08
anti normalize :  
[[ 1666.16796875]
 [ 1760.6394043 ]
 [ 1855.11108398]]

train_x shape :  (9, 3)
train_y shape :  (9, 1)
test_x shape :  (3, 3)
test_y shape :  (3, 1)
test_ y :  
[[ 1700.]
 [ 1800.]
 [ 1900.]]

 rmse  loss :  12.3943
"""