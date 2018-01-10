import tensorflow as tf 
import numpy as np 


# 加载预训练数据 
modelPath = "vgg19.npy"
wDict = np.load(modelPath, encoding = "bytes").item()

name='conv1_1'
p_w=wDict[name][0]
p_b=wDict[name][1]


with tf.variable_scope(name) as scope:
    # 将预训练的数据 变成tf里面的变量
    var1=tf.get_variable("w",shape=p_w.shape).assign(p_w)
    var2=tf.get_variable("b",shape=p_b.shape).assign(p_b)

    # resuse 变量 进行卷积运算
    scope.reuse_variables()
    w =tf.get_variable("w",shape=p_w.shape)
    b =tf.get_variable("b",shape=p_b.shape)

    # 注意输入的x 是一张图，维度是(224,224,3)，但是前面记得给他加上一个 (1，224,224,3)
    x = np.ones((1,224,224,3),dtype=np.float32)
    featuremap = tf.nn.conv2d(x,w,strides = [1, 1, 1, 1], padding = "SAME")
    out = tf.nn.bias_add(featuremap,b)


init = tf.global_variables_initializer() 
with tf.Session() as sess:
    sess.run(init)

    # [kHeight, kWidth, channel, featureNum] 
    # print ("p_w shape : ",p_w.shape)  #(3, 3, 3, 64)
    # print ("p_b shape : ",p_b.shape)
    print (var1.name," - ",sess.run(var1))
    print (var2.name," - ",sess.run(var2))
    print (" out : ",sess.run(out).shape)  # out :  (1, 224, 224, 64)