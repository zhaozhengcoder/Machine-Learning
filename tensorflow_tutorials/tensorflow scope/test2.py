import tensorflow as tf
import numpy as np


with tf.variable_scope('v_scope'):
    Weights1 = tf.get_variable('Weights', shape=[2,3])
    bias1 = tf.Variable([0.52], name='bias')

# resue 的作用是共享上面已经定义好的变量
# 下面来共享上面已经定义好的变量
# note: 在下面的 scope 中的get_variable()变量必须已经定义过了，才能设置 reuse=True，否则会报错
with tf.variable_scope('v_scope', reuse=True):
    Weights2 = tf.get_variable('Weights')
    bias2 = tf.Variable([0.52], name='bias')

init =tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print (Weights1.name)
    print (Weights2.name)