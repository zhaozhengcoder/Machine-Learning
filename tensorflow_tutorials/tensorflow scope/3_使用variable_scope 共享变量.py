import tensorflow as tf
import numpy as np


def my_image_file(input_images):
    conv1_weights = tf.get_variable("weights", [3,4],initializer=tf.random_normal_initializer())
    return conv1_weights


input1=tf.get_variable(name="var1", initializer=np.ones (shape=[2,3],dtype=np.float32))
input2=tf.get_variable(name="var2", initializer=np.zeros(shape=[2,3],dtype=np.float32))


with tf.variable_scope("image_filters") as scope:
    ret1 = my_image_file(input1)
    scope.reuse_variables()
    ret2 = my_image_file(input2)



init =tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print (ret1.name,sess.run(ret1))
    print (ret2.name,sess.run(ret2))

