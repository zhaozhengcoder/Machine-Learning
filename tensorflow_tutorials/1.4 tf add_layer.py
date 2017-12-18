import tensorflow as tf

#定义添加神经层的函数def add_layer(),它有四个参数：输入值、输入的大小、输出的大小和激励函数
#我们设定默认的激励函数是None
from __future__ import print_function
import tensorflow as tf

# 个人的看法，在每一层做计算的时候，要搞清楚矩阵的维度
# 这个的表示方式和吴恩达教程上面的表示方式是相反的
# inputs =[样本数 * 特征数] ，而吴恩达的教程是 特征数*样本数

"""
所以，这里的表示方式是： input * weights 
假如，输入层的结点个数是2，隐层是3
input=[n*2]  ,weihts=[2*3] ,bias=[1,3]
input*weigths=[n,3] + bias=[1,3] ，这样的矩阵维度相加的时候，python会执行它的广播机制

so,这一层的输出的维度是 [n,3]
"""
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs