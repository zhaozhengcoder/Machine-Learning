"""
构造一个3层的网络
输入层一个结点，隐层3个结点，输出层一个结点

输入层的维度是[n,1]
隐层的维度是  [1,10]
输出层的维度是[10,1]

so,
权值矩阵的维度是：
weight1=[1,10]
bais1=[10,1]

weight2=[10,1]
bais2=[1,1]
"""


import tensorflow as tf
import numpy as np

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

# 构造一个数据集
x_data = np.linspace(-1,1,300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# placeholder 占个位
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# add hidden layer
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)

# add output layer
# 上一层的输出是这一层的输入
prediction = add_layer(l1, 10, 1, activation_function=None)

# the error between prediction and real data
#loss函数和使用梯度下降的方式来求解
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12

if int((tf.__version__).split('.')[1]) < 12:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(1000):
    # trainings
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        # to see the step improvement
        # 在带有placeholder的变量里面，每一次sess.run 都需要给一个feed_dict，这个不能省略啊！
        print("loss : ",sess.run(loss, feed_dict={xs: x_data, ys: y_data}))