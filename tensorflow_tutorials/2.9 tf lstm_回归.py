
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pylab as plt
from tensorflow.contrib import rnn

def split_dataset(dataset,time_step):
    days,ndim = dataset.shape
    dataX=[]
    dataY=[]
    for i in range(0,days-time_step):
        dataX.append(dataset[i:i+time_step])
        dataY.append(dataset[i+time_step:i+time_step+1])
        #print ("x ",i," ",i+time_step)
        #print ("y ",i+time_step," ",i+time_step+1)
    return np.array(dataX),np.array(dataY)




dataset = np.random.rand(53,288)  # shape 53 * 288


# hyperparameters
batch = 50
n_inputs = 288
n_steps = 3             # time steps
n_hidden_units = 512    # neurons in hidden layer
n_output = 288
layer_num = 1
dropout_keep_rate = 0.9
max_epoch = int(2000 * 6)  # 6

# 归一化
scaler = MinMaxScaler(feature_range=(0,1))
dataset_scaler = scaler.fit_transform(dataset)

#
dataX,dataY = split_dataset(dataset_scaler,time_step=n_steps)  #dataX shape (50,3,288) ,dataY shape (50,1,288)
dataY=np.reshape(dataY,(batch,n_output))                       #dataY shape (50,1,288)




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




# 根据输入数据来决定，train_num训练集大小,input_size输入维度
#train_num, time_step_size, input_size = dataX.shape  # sahpe ：12 * 2 *480
#_, output_size = dataY.shape

# **步骤1：LSTM 的输入shape = (batch_size, time_step_size, input_size)，输出shape=(batch_size, output_size)
x_input = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y_real = tf.placeholder(tf.float32, [None, n_output])

# dropout的留下的神经元的比例
keep_prob = tf.placeholder(tf.float32, [])

# 在训练和测试的时候，我们想用不同的 batch_size.所以采用占位符的方式
batch_size = tf.placeholder(tf.int32, [])  # 注意类型必须为 tf.int32

pre_layer_hidden_num = 0
pre_layer_hidden_size = 0
hide_output = x_input

y_pred = lstm(layer_num, n_inputs, batch_size, n_output, hide_output, keep_prob)
mse = tf.losses.mean_squared_error(y_real, y_pred)
train_op = tf.train.AdamOptimizer(1e-3).minimize(mse)


sess = tf.Session()

# 初始化变量
sess.run(tf.global_variables_initializer())



for i in range(1, max_epoch + 1):
    feed_dict = {x_input: dataX, y_real: dataY, keep_prob: dropout_keep_rate, batch_size: batch}
    sess.run(train_op, feed_dict=feed_dict)
    print(sess.run(mse, feed_dict=feed_dict))

    if i % 500 == 0:
        y_pre = sess.run(y_pred, feed_dict=feed_dict)  # (50,288)
        plt.plot(y_pre[0])
        plt.plot(dataY[0])
        plt.show()
        plt.close()
