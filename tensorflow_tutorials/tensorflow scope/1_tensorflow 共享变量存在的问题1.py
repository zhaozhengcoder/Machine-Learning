import tensorflow as tf
import numpy as np


# demo1 
def my_image_file(input):
    conv1_weights = tf.Variable(tf.random_normal([3,4]),name="conv1_weights")
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
# demo1 的输出
$ python scope_test2.py
[[ 2.19378233  0.98282391 -0.02128156 -2.82178164]
 [-0.16895325 -1.37853122  0.57983387  0.52190787]
 [-0.93033606 -0.10535703 -0.44646293  0.49862373]]
[[-0.4303959  -0.37698492  0.23179258  0.81338549]
 [-1.65708673  0.03731534  1.02826369 -0.02426604]
 [-0.81754178  1.41695929  1.03476858  0.12911084]]


解释：
当我们想重用这个函数的时候，假设你想把你的图片过滤器运用到两张不同的图片image1和image2.
你想通过拥有同一个参数的同一个过滤器来过滤两张图片，你可以调用my_image_filter()两次，但是这会产生两组变量.
but
这种情况是我们不想要的，因为调用了两次，但是这两次每一次的权值是不一样的。

详情 ： http://wiki.jikexueyuan.com/project/tensorflow-zh/how_tos/variable_scope.html

这个情况，如何解决，就是通过共享变量的方式来解决的。
"""

