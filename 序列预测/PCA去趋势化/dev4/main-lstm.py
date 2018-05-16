import numpy as np
import pickle

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



name_count=0
def show1_ticks(arr_true,arr_prediction):
    #改变x轴的刻度
    x_kedu=[0,4,8,12,16,20,24]
    orig_ticks = [i*20 for i in x_kedu]
    new_ticks = x_kedu
    plt.xticks(orig_ticks,new_ticks)
    #改变y轴的刻度
    y_kedu=[0,5,10,15,20,25]
    y_orig_ticks = y_kedu
    y_new_ticks =y_kedu
    plt.yticks(y_orig_ticks,y_new_ticks)
    
    plt.xlabel("Time (hour)")
    plt.ylabel("Traffic volume")
    plt.plot(arr_true,'b-',label='true')
    plt.plot(arr_prediction,'r-',label='prediction')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()
    global name_count
    #plt.savefig(str(name_count)+".png")
    plt.close()
    name_count+=1


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

    #train_x_raw = trainX[0]                                       # 取了第一条路段来进行预测 train_x_raw的sahpe ：12 * 2 *480
    #train_y_raw = np.reshape(trainY[0], (days - time_step, 480))  # train_y_raw的shape 12 * 480
    train_x_raw = np.reshape(trainX, (road_num * (days - time_step),time_step,dnum))   #train_x_raw (20 * 12,2,480)
    train_y_raw = np.reshape(trainY, (road_num * (days - time_step),dnum))             #train_y_raw (20 * 12,1,480)

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

    test_x = train_x[train_len:]
    test_y = train_y[train_len:]
    #train_x =train_x[0:train_len]
    #train_y =train_y[0:train_len]
    train_x =train_x
    train_y =train_y

    # 放入lstm训练
    # lstm的hyper-parameter
    hidden_size = 200
    layer_num = 1
    max_epoch = int(2000 * 6)  #6
    dropout_keep_rate = 0.9

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

    # 设置 GPU 按需增长
    #config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    # config.gpu_options.allow_growth = True
    #sess = tf.Session(config)
    # cpu 
    sess = tf.Session()

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

    def cal_mre(y,y_pre):
        y[y < 0.1] = 0.1
        diff = np.abs(y-y_pre)
        mre = np.mean(diff/y)
        print("cal mre is : ", mre)

    def print_to_console(i, train_y ,train_y_pred,flag_istrain):
        train_y_pred_real = train_y_pred * (y_max - y_min) + y_min  # 反归一化
        train_y_real = train_y * (y_max - y_min) + y_min            # train_y 是真实的y值，堆train_y 进行反归一化

        #plt.plot(range(dnum), train_y_real[0], 'b-',label='true')        # 实际用蓝色
        #plt.plot(range(dnum), train_y_pred_real[0], 'r-',label='prediction')   # 预测用红色
        #plt.show()
        """
        if (flag_istrain==1):
            plt.savefig("train"+str(i)+".png")
        else:
            plt.savefig("test" +str(i)+".png")
        """
        train_mre, train_mae, train_rmse = get_metrics(train_y_real, train_y_pred_real)
        if(flag_istrain==1):
            #print("epoch {} train : {} {} {} ".format(i, train_mre, train_mae, train_rmse))
            print("epoch {} train : {} {} ".format(i, train_mre, train_mae))
        else:
            print("epoch {} train : {} {} ".format(i, train_mre, train_mae))
            #print("epoch {} test : {} {} {} ".format(i, train_mre, train_mae, train_rmse))

    def print_to_console_show(i, train_y ,train_y_pred,flag_istrain):
        train_y_pred_real = train_y_pred * (y_max - y_min) + y_min  # 反归一化
        train_y_real = train_y * (y_max - y_min) + y_min            # train_y 是真实的y值，堆train_y 进行反归一化
        plt.plot(range(dnum), train_y_real[0], 'b-',label='true')        # 实际用蓝色
        plt.plot(range(dnum), train_y_pred_real[0], 'r-',label='prediction')   # 预测用红色
        plt.show()
        """
        if (flag_istrain==1):
            plt.savefig("train"+str(i)+".png")
        else:
            plt.savefig("test" +str(i)+".png")
        """


    def cal_total_inlstm(y_raw_test,y_pre_test_real):  #true ,pred
        shape_len=7.0
        test_len = y_raw_test.shape[0]
        test_mre=0.0
        test_mae=0.0
        for i in range(0,test_len):
            res = get_metrics(y_raw_test[i],y_pre_test_real[i])
            test_mre+=res[0]
            test_mae+=res[1]
        #print ("2 : {} , {} ".format(test_mre/test_len,test_mae/(test_len+shape_len)))
        return test_mre/test_len,test_mae/(test_len+shape_len)


    for i in range(0, max_epoch + 1):
        feed_dict = {x_input: train_x, y_real: train_y, keep_prob: dropout_keep_rate, batch_size: train_num}
        sess.run(train_op, feed_dict=feed_dict)
        show_iter=50   #50
        if i % show_iter == 0:
            feed_dict = {x_input: train_x, y_real: train_y, keep_prob: 1.0, batch_size: train_num}
            train_y_pred = sess.run(y_pred, feed_dict=feed_dict)
            cost = sess.run(mse, feed_dict=feed_dict)
            print ("iter : ",i," ",cost )
            #print ("train_y_pred : ",train_y_pred.shape)    #(9,480)
            #print_to_console_show(i,train_y, train_y_pred,1)
        if i % show_iter ==0:
            feed_dict = {x_input: test_x, y_real: test_y, keep_prob: 1.0, batch_size: test_len}
            test_y_pred = sess.run(y_pred, feed_dict=feed_dict)
            #print ("test_y_pred : ",test_y_pred.shape)      #(3,480)
            #print_to_console_show(i, test_y,test_y_pred,0)
    
    #train
    feed_dict = {x_input: train_x, y_real: train_y, keep_prob: 1.0, batch_size: train_num}    
    train_y_pred = sess.run(y_pred, feed_dict=feed_dict)
    train_y_pred = np.reshape(train_y_pred,(road_num,int(days-time_step),dnum))
    train_y      = np.reshape(train_y,     (road_num,int(days-time_step),dnum))

    #test
    feed_dict = {x_input: test_x, y_real: test_y, keep_prob: 1.0, batch_size: test_len}
    test_y_pred = sess.run(y_pred, feed_dict=feed_dict)
    test_y_pred = np.reshape(test_y_pred,  (road_num,int(days*test_split_rate),dnum))  #(20,3,480)
    test_y = np.reshape(test_y,(road_num,int(days*test_split_rate),dnum))            #(20,3,480)
    
    # 反归一化
    test_y_pred = test_y_pred* (y_max - y_min) + y_min
    test_y = test_y * (y_max - y_min) + y_min
    train_y_pred = train_y_pred* (y_max - y_min) + y_min
    train_y = train_y * (y_max - y_min) + y_min
    
    #plt.plot(range(dnum), test_y[0][0], 'b-',label='true')              # 实际用蓝色
    #plt.plot(range(dnum), test_y_pred[0][0], 'r-',label='prediction')   # 预测用红色
    #plt.legend(loc='upper right')
    #plt.show()
    show1_ticks(train_y[0][0],train_y_pred[0][0])
    show1_ticks(train_y[0][1],train_y_pred[0][1])
    show1_ticks(train_y[0][2],train_y_pred[0][2])

    show1_ticks(test_y[0][0],test_y_pred[0][0])
    show1_ticks(test_y[0][1],test_y_pred[0][1])
    show1_ticks(test_y[0][2],test_y_pred[0][2])
    
    ret1 = cal_total_inlstm(test_y,test_y_pred)
    print("mre, mae  : ",ret1[0],"  ",ret1[1])

"""
iter :  0   0.026764797
iter :  50   0.0012212617
iter :  100   0.00096361485
iter :  150   0.00083238096
iter :  200   0.0007910932
iter :  250   0.0007242314
iter :  300   0.0006987227
iter :  350   0.0006629278
iter :  400   0.0006410195
iter :  450   0.00060583884
iter :  500   0.0006586664
iter :  550   0.0005667903
iter :  600   0.0005869594
iter :  650   0.00052952924
iter :  700   0.00050735445
iter :  750   0.00050338346
iter :  800   0.0004978407
iter :  850   0.00047489488
iter :  900   0.00045653927
iter :  950   0.0004565282
iter :  1000   0.0004400905
iter :  1050   0.00045980222
iter :  1100   0.00040708628
iter :  1150   0.00039873447
iter :  1200   0.00038895616
iter :  1250   0.00039470228
iter :  1300   0.0003712386
iter :  1350   0.00038502782
iter :  1400   0.00035337068
iter :  1450   0.000340099
iter :  1500   0.0003361163
iter :  1550   0.00037948496
iter :  1600   0.0003356651
iter :  1650   0.0003302114
iter :  1700   0.00030387705
iter :  1750   0.00030219456
iter :  1800   0.00029587458
iter :  1850   0.00027538426
iter :  1900   0.00031056252
iter :  1950   0.00026644685
iter :  2000   0.00027223353
iter :  2050   0.00029456985
iter :  2100   0.00026179836
iter :  2150   0.00023973371
iter :  2200   0.00023161515
iter :  2250   0.00022887727
iter :  2300   0.00022530508
iter :  2350   0.00022560568
iter :  2400   0.00022864623
iter :  2450   0.00020548009
iter :  2500   0.00020977242
iter :  2550   0.00019409544
iter :  2600   0.00021064942
iter :  2650   0.0001907086
iter :  2700   0.00019929222
iter :  2750   0.0001795655
iter :  2800   0.00017285744
iter :  2850   0.0001721027
iter :  2900   0.00016609495
iter :  2950   0.00017817848
iter :  3000   0.00017265706
iter :  3050   0.00016325891
iter :  3100   0.00015106231
iter :  3150   0.00014628579
iter :  3200   0.00014172103
iter :  3250   0.00014043265
iter :  3300   0.00014535316
iter :  3350   0.00013400015
iter :  3400   0.00013404756
iter :  3450   0.00012781448
iter :  3500   0.00012982085
iter :  3550   0.00013350998
iter :  3600   0.00012925516
iter :  3650   0.000120690805
iter :  3700   0.000112850634
iter :  3750   0.000116543015
iter :  3800   0.000111889494
iter :  3850   0.000106518186
iter :  3900   0.00011482349
iter :  3950   0.00012147474
iter :  4000   0.00010801501
iter :  4050   0.000102698636
iter :  4100   0.000100229845
iter :  4150   9.854034e-05
iter :  4200   9.98105e-05
iter :  4250   9.691938e-05
iter :  4300   9.450808e-05
iter :  4350   9.15536e-05
iter :  4400   9.320474e-05
iter :  4450   9.0609836e-05
iter :  4500   8.7422064e-05
iter :  4550   8.386313e-05
iter :  4600   9.1314236e-05
iter :  4650   8.690134e-05
iter :  4700   9.0412235e-05
iter :  4750   8.189807e-05
iter :  4800   7.625681e-05
iter :  4850   8.622256e-05
iter :  4900   8.292348e-05
iter :  4950   7.2486e-05
iter :  5000   7.4743744e-05
iter :  5050   7.187687e-05
iter :  5100   7.13506e-05
iter :  5150   7.097006e-05
iter :  5200   6.776469e-05
iter :  5250   7.0884504e-05
iter :  5300   7.0944196e-05
iter :  5350   6.913516e-05
iter :  5400   7.141812e-05
iter :  5450   6.566111e-05
iter :  5500   7.787771e-05
iter :  5550   6.182677e-05
iter :  5600   6.414938e-05
iter :  5650   6.9835565e-05
iter :  5700   6.562619e-05
iter :  5750   6.4331754e-05
iter :  5800   5.8664853e-05
iter :  5850   5.8527963e-05
iter :  5900   5.9268725e-05
iter :  5950   5.922038e-05
iter :  6000   5.9299648e-05
iter :  6050   6.173521e-05
iter :  6100   6.297361e-05
iter :  6150   5.8630132e-05
iter :  6200   6.0070357e-05
iter :  6250   7.220554e-05
iter :  6300   5.1195184e-05
iter :  6350   5.394911e-05
iter :  6400   5.789791e-05
iter :  6450   5.2303516e-05
iter :  6500   5.5177763e-05
iter :  6550   5.8093956e-05
iter :  6600   5.784504e-05
iter :  6650   5.713607e-05
iter :  6700   5.663139e-05
iter :  6750   5.197638e-05
iter :  6800   4.786879e-05
iter :  6850   5.3402484e-05
iter :  6900   4.9448907e-05
iter :  6950   4.8961276e-05
iter :  7000   4.9565668e-05
iter :  7050   5.502768e-05
iter :  7100   5.156186e-05
iter :  7150   5.032753e-05
iter :  7200   4.9621074e-05
iter :  7250   5.5263085e-05
iter :  7300   5.2352076e-05
iter :  7350   4.8564034e-05
iter :  7400   4.6030018e-05
iter :  7450   4.4059194e-05
iter :  7500   4.5506724e-05
iter :  7550   4.3649714e-05
iter :  7600   4.754362e-05
iter :  7650   4.60165e-05
iter :  7700   4.1899613e-05
iter :  7750   4.398889e-05
iter :  7800   4.5359306e-05
iter :  7850   4.1706357e-05
iter :  7900   4.485839e-05
iter :  7950   4.4016095e-05
iter :  8000   4.5656612e-05
iter :  8050   4.3595042e-05
iter :  8100   4.0542585e-05
iter :  8150   4.124076e-05
iter :  8200   4.4175802e-05
iter :  8250   4.4820557e-05
iter :  8300   3.8113743e-05
iter :  8350   5.5513825e-05
iter :  8400   4.1448926e-05
iter :  8450   4.707205e-05
iter :  8500   4.871493e-05
iter :  8550   4.173235e-05
iter :  8600   4.1141415e-05
iter :  8650   5.3598185e-05
iter :  8700   4.6822628e-05
iter :  8750   3.9408777e-05
iter :  8800   4.3088952e-05
iter :  8850   4.6244822e-05
iter :  8900   3.822931e-05
iter :  8950   4.1194882e-05
iter :  9000   3.9517887e-05
iter :  9050   4.7127895e-05
iter :  9100   4.2782656e-05
iter :  9150   4.234408e-05
iter :  9200   3.5613244e-05
iter :  9250   3.9477673e-05
iter :  9300   3.8048584e-05
iter :  9350   3.814626e-05
iter :  9400   4.1246967e-05
iter :  9450   3.989246e-05
iter :  9500   4.5052813e-05
iter :  9550   3.6963134e-05
iter :  9600   3.9859904e-05
iter :  9650   3.7005724e-05
iter :  9700   3.9210507e-05
iter :  9750   4.7843434e-05
iter :  9800   3.534675e-05
iter :  9850   3.4058798e-05
iter :  9900   3.707529e-05
iter :  9950   3.61862e-05
iter :  10000   3.7150905e-05
iter :  10050   3.5458517e-05
iter :  10100   3.6721056e-05
iter :  10150   3.402754e-05
iter :  10200   3.4949448e-05
iter :  10250   4.1215717e-05
iter :  10300   3.6490324e-05
iter :  10350   3.5818884e-05
iter :  10400   4.7231457e-05
iter :  10450   3.4786575e-05
iter :  10500   4.1003335e-05
iter :  10550   3.840775e-05
iter :  10600   3.8680555e-05
iter :  10650   4.131663e-05
iter :  10700   3.911375e-05
iter :  10750   3.5705234e-05
iter :  10800   3.5845464e-05
iter :  10850   3.434616e-05
iter :  10900   3.5316414e-05
iter :  10950   3.560815e-05
iter :  11000   4.2883614e-05
iter :  11050   3.4726803e-05
iter :  11100   4.83908e-05
iter :  11150   3.5482e-05
iter :  11200   3.734082e-05
iter :  11250   3.582418e-05
iter :  11300   3.445597e-05
iter :  11350   3.7683843e-05
iter :  11400   3.4743043e-05
iter :  11450   3.5619534e-05
iter :  11500   3.234973e-05
iter :  11550   3.6054258e-05
iter :  11600   3.5540812e-05
iter :  11650   3.5944584e-05
iter :  11700   3.5310084e-05
iter :  11750   3.03455e-05
iter :  11800   3.4035103e-05
iter :  11850   3.442646e-05
iter :  11900   3.5994846e-05
iter :  11950   3.7213445e-05
iter :  12000   3.2510394e-05
mre, mae  :  0.12447681647919173    0.44161080753467585
"""