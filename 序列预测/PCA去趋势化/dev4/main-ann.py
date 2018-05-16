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




if __name__ == "__main__":
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

    #ANN
    xs=tf.placeholder(tf.float32,[None,960])
    ys=tf.placeholder(tf.float32,[None,480])

    l1=add_layer(xs,960,1500,activation_function=tf.nn.sigmoid)
    l2=add_layer(l1,1500,1000,activation_function=tf.nn.sigmoid)
    prediction=add_layer(l2,1000,480,activation_function=None)

    #loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
    #train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    mse = tf.losses.mean_squared_error(ys, prediction)
    print (mse)
    train_op = tf.train.AdamOptimizer(0.001).minimize(mse)

    if int((tf.__version__).split('.')[1]) < 12:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    
    sess = tf.Session()
    sess.run(init)

    def print_to_console(i, train_y ,train_y_pred,flag_istrain):
        train_y_pred_real = train_y_pred * (y_max - y_min) + y_min  # 反归一化
        train_y_real = train_y * (y_max - y_min) + y_min            # train_y 是真实的y值，堆train_y 进行反归一化

        plt.plot(range(dnum), train_y_real[0], 'b-',label='true')        # 实际用蓝色
        plt.plot(range(dnum), train_y_pred_real[0], 'r-',label='prediction')   # 预测用红色
        plt.legend(loc='upper right')
        if (flag_istrain==1):
            plt.savefig("train"+str(i)+".png")
        else:
            plt.savefig("test" +str(i)+".png")
        plt.close()

        train_mre, train_mae, train_rmse = get_metrics(train_y_real, train_y_pred_real)
        if(flag_istrain==1):
            #print("epoch {} train : {} {} {} ".format(i, train_mre, train_mae, train_rmse))
            print("epoch {} train : {} {} ".format(i, train_mre, train_mae))
        else:
            #print("epoch {} test : {} {} {} ".format(i, train_mre, train_mae, train_rmse))
            print("epoch {}  : {} {} ".format(i, train_mre, train_mae))


    for i in range(25000):
        # trainings
        sess.run(train_op, feed_dict={xs: train_x, ys: train_y})
        #print ("mse " , sess.run(mse, feed_dict={xs: train_x, ys: train_y}))
        iter = 1000
        if i % iter == 0:
            # to see the step improvement
            # 在带有placeholder的变量里面，每一次sess.run 都需要给一个feed_dict，这个不能省略啊！
            print ("cost : ",sess.run(mse, feed_dict={xs: train_x, ys: train_y}))
            train_y_pred = sess.run(prediction, feed_dict={xs: train_x, ys: train_y})
            #print ("shape : ",train_pre.shape)
            #print ("mre, mae, rmse : " , get_metrics(train_y,train_y_pred))
            #print_to_console(i,train_y, train_y_pred,1)
        if i % iter == 0:
            #print ("cost : ",sess.run(mse, feed_dict={xs: test_x, ys: test_y}))
            test_y_pred = sess.run(prediction, feed_dict={xs: test_x, ys: test_y})
            print_to_console(i,test_y, test_y_pred,0)