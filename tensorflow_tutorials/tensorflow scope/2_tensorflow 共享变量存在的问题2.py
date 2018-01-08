import tensorflow as tf
import numpy as np

# 使用字典的方式实现共享变量

# demo2 
variables_dict = {
    "conv1_weights":tf.Variable(tf.random_normal([2,3]),name="conv1_weights"),
    "conv1_biases":tf.Variable(tf.zeros([5]), name="conv1_biases")
}

def my_image_file(input):
    conv1_weights = variables_dict['conv1_weights']
    return conv1_weights

input1=tf.get_variable(name="var1", initializer=np.ones (shape=[2,3],dtype=np.float32))
input2=tf.get_variable(name="var2", initializer=np.zeros(shape=[2,3],dtype=np.float32))

ret1=my_image_file(input1)
ret2=my_image_file(input2)

init =tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print (sess.run(ret1))
    print (sess.run(ret2))


"""
#demo2 的输出
$ python scope_test2.py
[[-0.72093755  0.70740443 -0.10332351]
 [ 0.43170732 -0.01485203  0.39022416]]
[[-0.72093755  0.70740443 -0.10332351]
 [ 0.43170732 -0.01485203  0.39022416]]

这个通过字典的方式，来解决共享变量的方式来搞定的。
"""

