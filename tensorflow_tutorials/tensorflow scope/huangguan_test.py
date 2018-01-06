import numpy as np
import tensorflow as tf

x = np.ones([10, 5], dtype=np.float32)
y = np.zeros([20, 5], dtype=np.float32)
# xp = tf.placeholder(dtype=tf.float32, shape=[None, 5])
# y_pre = tf.layers.dense(xp, 2)
with tf.variable_scope('xx'):
    x1 = tf.get_variable('xt', initializer=np.ones(shape=[5, 10], dtype=np.float32))

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    x2 = tf.get_variable('xt2', shape=[10, 5], dtype=tf.float32).assign(x)
    x3 = tf.matmul(x1, x2)
    xx3 = sess.run(x3)
    print(xx3)
