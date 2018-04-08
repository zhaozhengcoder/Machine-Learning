import numpy as np
import pickle
import pca
import tensorflow as tf
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt


def lstm(layer_num, hidden_size, batch_size, output_size, lstm_x, keep_prob):
    def multi_cells(cell_num):
        # 多cell的lstm必须多次建立cell保存在一个list当中
        multi_cell = []
        for _ in range(cell_num):
            # **步骤2：定义LSTM_cell，只需要说明 hidden_size, 它会自动匹配输入的 X 的维度
            lstm_cell = rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)

            # **步骤3：添加 dropout layer, 一般只设置 output_keep_prob
            lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
            multi_cell.append(lstm_cell)
        return multi_cell

    # **步骤4：调用 MultiRNNCell 来实现多层 LSTM
    mlstm_cell = rnn.MultiRNNCell(multi_cells(layer_num), state_is_tuple=True)

    # **步骤5：用全零来初始化state
    init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)

    # **步骤6：调用 dynamic_rnn() 来让我们构建好的网络运行起来
    # ** 当 time_major==False 时， outputs.shape = [batch_size, time_step_size, hidden_size]
    # ** 所以，可以取 h_state = outputs[:, -1, :] 作为最后输出
    # ** state.shape = [layer_num, 2, batch_size, hidden_size]（中间的‘2’指的是每个cell中有两层分别是c和h）,
    # ** 或者，可以取 h_state = state[-1][1] 作为最后输出
    # ** 最后输出维度是 [batch_size, hidden_size]
    outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=lstm_x, initial_state=init_state, time_major=False)
    h_state = outputs[:, -1, :]  # 或者 h_state = state[-1][1]

    # 输出层
    # W_o = tf.Variable(tf.truncated_normal([hidden_size, output_size], stddev=0.1), dtype=tf.float32)
    # b_o = tf.Variable(tf.constant(0.1, shape=[output_size]), dtype=tf.float32)
    # y_pre = tf.add(tf.matmul(h_state, W_o), b_o)
    # tf.layers.dense是全连接层，不给激活函数，默认是linear function
    lstm_y_pres = tf.layers.dense(h_state, output_size)
    return lstm_y_pres


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
    f = open('data/' + filename, 'rb')
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


def use_pca(dataset):
    dataset_rest = []
    dataset_main = []
    for road in dataset:
        pca_obj = pca.PCA(road, 4)
        dataset_rest.append(pca_obj.rest_x)
        dataset_main.append(pca_obj.main_x)
    return dataset_main, dataset_rest


if __name__ == "__main__":
    time_step = 2
    data = myload()  # data是一个list，里面是df格式的数据
    # transfer
    dataset = createdataset(data)  # dataset 的格式是 （路段 * 每一天 * 一天内的数据）20 * 7 * 480
    dataset_main, dataset_rest = use_pca(dataset)  # dataset_main = 路段 * 每一个路段里面的主成分 ; dataset_rest = 路段 * 每一个路段里面的偏差
    days, dnum = dataset_main[0].shape
    y_main = dataset_main[0][time_step:days, :]
    trainX, trainY = split_dataset(dataset_rest)
    train_x_raw = trainX[0]
    train_y_raw = np.reshape(trainY[0], (days - time_step, 480))

    x_max = train_x_raw.max()
    x_min = train_x_raw.min()
    y_max = train_y_raw.max()
    y_min = train_y_raw.min()

    train_x = (train_x_raw - x_min) / (x_max - x_min)
    train_y = (train_y_raw - y_min) / (y_max - y_min)

    # lstm的hyper-parameter

    hidden_size = 400
    layer_num = 1
    max_epoch = 5000
    dropout_keep_rate = 1

    # 根据输入数据来决定，train_num训练集大小,input_size输入维度
    train_num, time_step_size, input_size = train_x.shape
    # output_size输出的结点个数
    _, output_size = train_y.shape

    # **步骤1：LSTM 的输入shape = (batch_size, time_step_size, input_size)，输出shape=(batch_size, output_size)
    x_input = tf.placeholder(tf.float32, [None, time_step_size, input_size])
    y_real = tf.placeholder(tf.float32, [None, output_size])

    # dropout的留下的神经元的比例
    keep_prob = tf.placeholder(tf.float32, [])

    # 在训练和测试的时候，我们想用不同的 batch_size.所以采用占位符的方式
    batch_size = tf.placeholder(tf.int32, [])  # 注意类型必须为 tf.int32

    pre_layer_hidden_num = 0
    pre_layer_hidden_size = 0
    hide_output = x_input

    y_pred = lstm(layer_num, hidden_size, batch_size, output_size, hide_output, keep_prob)
    # 损失和评估函数
    mse = tf.losses.mean_squared_error(y_real, y_pred)
    train_op = tf.train.AdamOptimizer(1e-3).minimize(mse)

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    # 设置 GPU 按需增长
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    # 初始化变量
    sess.run(tf.global_variables_initializer())

    mre_result = []
    mae_result = []
    rmse_result = []


    def get_metrics(y, pred_y):
        mre = np.mean(np.abs(y - pred_y) / y)
        mae = np.mean(np.abs(y - pred_y))
        rmse = np.sqrt(np.mean(np.square(y - pred_y)))
        return mre, mae, rmse


    def print_to_console(i, train_y_pred):
        train_y_pred_real = train_y_pred * (y_max - y_min) + y_min
        train_y_real = train_y * (y_max - y_min) + y_min
        plt.plot(range(dnum), train_y_real[0], 'b-')
        plt.plot(range(dnum), train_y_pred_real[0], 'r-')
        plt.show()
        train_mre, train_mae, train_rmse = get_metrics(train_y_real, train_y_pred_real)
        print('epoch %d  train %.4f %.2f %.2f' % (i, train_mre, train_mae, train_rmse))


    for i in range(1, max_epoch + 1):
        feed_dict = {x_input: train_x, y_real: train_y, keep_prob: dropout_keep_rate, batch_size: train_num}
        sess.run(train_op, feed_dict=feed_dict)
        if i % 50 == 0:
            feed_dict = {x_input: train_x, y_real: train_y, keep_prob: 1.0, batch_size: train_num}
            train_y_pred = sess.run(y_pred, feed_dict=feed_dict)
            print_to_console(i, train_y_pred)

    y_main = dataset_main[0][time_step:days, :]
    y_pre_real = y_main + train_y_pred * (y_max - y_min) + y_min
    y_raw = y_main + train_y * (y_max - y_min) + y_min
    plt.plot(y_raw[0])
    plt.plot(y_pre_real[0])
    plt.show()
