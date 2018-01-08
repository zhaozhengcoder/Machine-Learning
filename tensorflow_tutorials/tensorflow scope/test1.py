import tensorflow as tf
import numpy as np

"""
with tf.name_scope('conv1') as scope:
    weights1 = tf.Variable([1.0, 2.0], name='weights')
    bias1 = tf.Variable([0.3], name='bias')


# 下面是在另外一个命名空间来定义变量的
with tf.name_scope('conv2') as scope:
    weights2 = tf.Variable([4.0, 2.0], name='weights')
    bias2 = tf.Variable([0.33], name='bias')


init =tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print (weights1.name)
    print (weights2.name)
"""

with tf.name_scope('conv1') as scope:
    weights1 = tf.Variable([1.0, 2.0], name='weights')
    bias1 = tf.Variable([0.3], name='bias')

with tf.name_scope('conv2') as scope:
    weights2 = tf.Variable([4.0, 2.0], name='weights')
    bias2 = tf.Variable([0.33], name='bias')


init =tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print (weights1.name)
    print (weights2.name)