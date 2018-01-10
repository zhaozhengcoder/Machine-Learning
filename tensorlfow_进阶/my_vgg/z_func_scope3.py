"""
一个基本的模型，可以看懂一下，后面就比较好理解了
"""

import tensorflow as tf 
import numpy as np 

modelPath = "vgg19.npy"
wDict = np.load(modelPath,encoding = "bytes").item()

def convLayer():
    x = np.ones((1,224,224,3),dtype=np.float32)
    #with tf.variable_scope(name) as scope:
    name='conv1_1'
    p_w=wDict[name][0]
    p_b=wDict[name][1]
    with tf.variable_scope(name):
        #scope.reuse_variables()
        w =tf.get_variable("w",shape=p_w.shape)
        b =tf.get_variable("b",shape=p_b.shape)
        featuremap = tf.nn.conv2d(x,w,strides = [1, 1, 1, 1], padding = "SAME")
        out = tf.nn.bias_add(featuremap,b)
        return out



def loadModel(self):
    """load model"""
    # 加载预训练数据 
    #for layers in model
    for name in wDict:
        with tf.variable_scope(name ,reuse =True):
            for p in wDict[name]:
                if len(p.shape) == 1:
                    #bias
                    tf.get_variable('b', trainable = False).assign(p)
                else:
                    #weights
                    tf.get_variable('w', trainable = False).assign(p)


out = convLayer()


init = tf.global_variables_initializer() 
with tf.Session() as sess:
    sess.run(init)
    
    #print (sess.run(out))

    # [kHeight, kWidth, channel, featureNum] 
    # print ("p_w shape : ",p_w.shape)  #(3, 3, 3, 64)
    # print ("p_b shape : ",p_b.shape)
    #print (var1.name," - ",sess.run(var1))
    #print (var2.name," - ",sess.run(var2))

    print (" out : ",sess.run(out).shape)  # out :  (1, 224, 224, 64)

    

    


