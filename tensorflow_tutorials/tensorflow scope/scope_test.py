import tensorflow as tf
import numpy as np

"""
# 使用tf.Variable 创建变量
var1=tf.Variable(name="var1",initial_value=[100],dtype=tf.float32)
var2=tf.Variable(name="var2",initial_value=[200],dtype=tf.float32)

init =tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print (var1.name," - ",sess.run(var1))
    print (var2.name," - ",sess.run(var2))

"""

"""
# 使用tf.get_variable 创建变量

var1=tf.get_variable(name="var1", initializer=np.ones (shape=[2, 3], dtype=np.float32))
var2=tf.get_variable(name="var2", initializer=np.zeros(shape=[2,3], dtype=np.float32))
init =tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print (var1.name," - ",sess.run(var1))
    print (var2.name," - ",sess.run(var2))
"""

"""
# 如果使用 tf.get_variable 创建两个相同名字的变量 ，就会报错
var1=tf.get_variable(name="var1", initializer=np.ones (shape=[2, 3], dtype=np.float32))
var2=tf.get_variable(name="var1", initializer=np.zeros(shape=[2,3], dtype=np.float32))
init =tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print (var1.name," - ",sess.run(var1))
    print (var2.name," - ",sess.run(var2))
"""

"""
# 即使使用name_scope ，get_variable的方式创建同名的变量也是会报错的
with tf.name_scope("a_name_scope"):
    var1=tf.get_variable(name="var1", initializer=np.ones (shape=[2, 3], dtype=np.float32))
    var2=tf.get_variable(name="var1", initializer=np.zeros(shape=[2,3], dtype=np.float32))

init =tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print (var1.name," - ",sess.run(var1))
    print (var2.name," - ",sess.run(var2))
"""


"""
# 使用tf.Variable 创建两个同名的变量，就不会报错！
# 使用tf.Variable 即使是同名的变量，也可以避免
var1=tf.Variable(name="var1",initial_value=[100],dtype=tf.float32)
var2=tf.Variable(name="var1",initial_value=[200],dtype=tf.float32)

init =tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print (var1.name," - ",sess.run(var1))
    print (var2.name," - ",sess.run(var2))

# 输出
# var1:0  -  [ 100.]
# var1_1:0  -  [ 200.]
"""

"""
# variable_scope 的理解
# var2 可以重新使用var1，也就是说var2就是var1 
# variable_scope 可以和 get_variable 结合使用
with tf.variable_scope("a_variable_scope") as scope:
    var1=tf.get_variable(name="var1", initializer=np.ones (shape=[2, 3], dtype=np.float32))
    # 这一句很关键，如果没有这个就报错了 
    scope.reuse_variables()
    var2=tf.get_variable(name="var1",)

init =tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print (var1.name," - ",sess.run(var1))
    print (var2.name," - ",sess.run(var2))

# a_variable_scope/var1:0  -  [[ 1.  1.  1.]
#  [ 1.  1.  1.]]
# a_variable_scope/var1:0  -  [[ 1.  1.  1.]
#  [ 1.  1.  1.]]
"""





