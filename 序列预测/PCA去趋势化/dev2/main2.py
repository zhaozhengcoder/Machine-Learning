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


def use_pca(dataset):
    dataset_rest = []
    dataset_main = []
    for road in dataset:
        pca_obj = pca.PCA(road, 2)
        dataset_rest.append(pca_obj.rest_x)
        dataset_main.append(pca_obj.main_x)
    return dataset_main, dataset_rest


if __name__ == "__main__":
    time_step = 2
    data = myload()  # data是一个list，里面是df格式的数据
    # transfer
    dataset = createdataset(data)                  # dataset 的格式是 （路段 * 每一天 * 一天内的数据）20 * 14 * 480
    dataset_main, dataset_rest = use_pca(dataset)  # dataset_main = 路段 * 每一个路段里面的主成分 ; dataset_rest = 路段 * 每一个路段里面的偏差
    days, dnum = dataset_main[0].shape

    y_main = dataset_main[0][time_step:days, :]    # y_main的主成分[2~14] shape 12 * 2 * 480
    #将偏差数据 分割 成输入和输出
    trainX, trainY = split_dataset(dataset_rest)   #trainX 的shape 20 * 12 * 2 *480 ； trainY 的shape 20 * 12 * 1 *480

    train_x_raw = trainX[0]  #取了第一条路段来进行预测 train_x_raw的sahpe ：12 * 2 *480
    train_y_raw = np.reshape(trainY[0], (days - time_step, 480))  # train_y_raw的shape 12 * 480
    # 归一化
    x_max = train_x_raw.max()
    x_min = train_x_raw.min()
    y_max = train_y_raw.max()
    y_min = train_y_raw.min()

    train_x = (train_x_raw - x_min) / (x_max - x_min)
    train_y = (train_y_raw - y_min) / (y_max - y_min)

    # 构造训练和测试集
    total_len = train_x.shape[0]
    train_len = int(total_len * 0.75)
    test_len = int(total_len *0.25)

    test_x = train_x[train_len:]
    test_y = train_y[train_len:]
    #train_x =train_x[0:train_len]
    #train_y =train_y[0:train_len]

    # 放入lstm训练
    # lstm的hyper-parameter
    hidden_size = 400
    layer_num = 1
    max_epoch = int(3000)
    dropout_keep_rate = 0.7

    # 根据输入数据来决定，train_num训练集大小,input_size输入维度
    train_num, time_step_size, input_size = train_x.shape     # sahpe ：12 * 2 *480
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

    # 获得训练的指标
    def get_metrics(y, pred_y):
        #替换为0的元素
        y_mean=np.mean(y)
        y[y==0.00] =y_mean
        mre = np.mean(np.abs(y - pred_y) / y)
        mae = np.mean(np.abs(y - pred_y))
        rmse = np.sqrt(np.mean(np.square(y - pred_y)))
        return mre, mae, rmse


    def print_to_console(i, train_y ,train_y_pred,flag_istrain,isshow):
        train_y_pred_real = train_y_pred * (y_max - y_min) + y_min  # 反归一化
        train_y_real = train_y * (y_max - y_min) + y_min            # train_y 是真实的y值，堆train_y 进行反归一化

        if(isshow==1):
            plt.plot(range(dnum), train_y_real[0], 'b-',label='true')        # 实际用蓝色
            plt.plot(range(dnum), train_y_pred_real[0], 'r-',label='prediction')   # 预测用红色
            plt.show()
        """
        if (flag_istrain==1):
            plt.savefig("train"+str(i)+".png")
        else:
            plt.savefig("test" +str(i)+".png")
        """
        train_mre, train_mae, train_rmse = get_metrics(train_y_real, train_y_pred_real)
        if(flag_istrain==1):
            #print('epoch %d  train :  %.4f %.2f %.2f' % (i, train_mre, train_mae, train_rmse))
            print("epoch {} train : {} {} {} ".format(i, train_mre, train_mae, train_rmse))
        else:
            #print('epoch %d  test :  %.4f %.2f %.2f' % (i, train_mre, train_mae, train_rmse))
            print("epoch {} test : {} {} {} ".format(i, train_mre, train_mae, train_rmse))

    for i in range(1, max_epoch + 1):
        feed_dict = {x_input: train_x, y_real: train_y, keep_prob: dropout_keep_rate, batch_size: train_num}
        sess.run(train_op, feed_dict=feed_dict)
        if i % 200 == 0:
            feed_dict = {x_input: train_x, y_real: train_y, keep_prob: 1.0, batch_size: train_num}
            train_y_pred = sess.run(y_pred, feed_dict=feed_dict)
            #print ("train_y_pred : ",train_y_pred.shape)    #(9,480)
            print_to_console(i,train_y, train_y_pred,1,0)
        if i % 200 ==0:
            feed_dict = {x_input: test_x, y_real: test_y, keep_prob: 1.0, batch_size: test_len}
            test_y_pred = sess.run(y_pred, feed_dict=feed_dict)
            #print ("test_y_pred : ",test_y_pred.shape)      #(3,480)
            print_to_console(i, test_y,test_y_pred,0,0)

    #to-do
    y_main = dataset_main[0][time_step:days, :]    # y_main的主成分[2~14] shape 12 * 2 * 480

    y_pre_train_real = y_main + train_y_pred * (y_max - y_min) + y_min   #train_y_pred的shape是 ：(9,480)
    y_raw_train = y_main + train_y * (y_max - y_min) + y_min   # true
    plt.plot(y_raw_train[0])  #只画第一天
    plt.plot(y_pre_train_real[0])
    plt.show()
    for i in range(0,train_len):
        print("train mre, mae, rmse : ", get_metrics(y_pre_train_real[i], y_raw_train[i]))


    y_pre_test_real = y_main[train_len:] + test_y_pred * (y_max - y_min) + y_min
    y_raw_test = y_main[train_len:] + test_y * (y_max - y_min) + y_min   # true
    plt.plot(y_pre_test_real[0],color='blue',label='prediction')  #只画第一天
    plt.plot(y_raw_test[0], color='red', label='true')
    plt.legend(loc='upper right')
    for i in range(0,test_len):
        print("test ",i,"  mre, mae, rmse : ", get_metrics(y_pre_test_real[i],y_raw_test[i]))
    plt.show()

### 大功告成 ！！！！
"""
train mre, mae, rmse :  (-0.029828510559539646, 0.029079688831018635, 0.03781274442185763)
train mre, mae, rmse :  (-0.014006342486120728, 0.029752569218725298, 0.03840164135038352)
train mre, mae, rmse :  (0.01966588730859742, 0.030049372790911334, 0.03848749067185995)
train mre, mae, rmse :  (0.0022712513481526657, 0.030908468525215952, 0.03962029519589479)
train mre, mae, rmse :  (-0.047443900425408074, 0.025280013859965516, 0.03241163721057964)
train mre, mae, rmse :  (-0.00021344378219514127, 0.030201234940533044, 0.03853920047170086)
train mre, mae, rmse :  (-0.0031145057760686525, 0.026909792760785048, 0.03449162880054893)
train mre, mae, rmse :  (-0.013424594270680072, 0.02744302404183101, 0.03515400456891561)
train mre, mae, rmse :  (-0.01507840636875168, 0.026399693560995384, 0.0340099252141404)
test  0   mre, mae, rmse :  (0.5089787653203184, 1.363375362604115, 1.8570183656086996)
test  1   mre, mae, rmse :  (-0.9159883432603605, 1.2817998982332992, 1.7140510132508648)
test  2   mre, mae, rmse :  (0.44473312781264535, 0.9941239026877812, 1.343086838355212)
"""