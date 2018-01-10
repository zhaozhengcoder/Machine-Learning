# coding: UTF-8
'''''''''''''''''''''''''''''''''''''''''''''''''''''
   file name: vgg19.py
   create time: 2017年04月13日 星期四 10时40分26秒
   author: Jipeng Huang
   e-mail: huangjipengnju@gmail.com
   github: https://github.com/hjptriplebee
'''''''''''''''''''''''''''''''''''''''''''''''''''''
import tensorflow as tf
import numpy as np
# define different layer functions
# we usually don't do convolution and pooling on batch and channel
def maxPoolLayer(x, kHeight, kWidth, strideX, strideY, name, padding = "SAME"):
    """max-pooling"""
    return tf.nn.max_pool(x, ksize = [1, kHeight, kWidth, 1],
                          strides = [1, strideX, strideY, 1], padding = padding, name = name)

def dropout(x, keepPro, name = None):
    """dropout"""
    return tf.nn.dropout(x, keepPro, name)

def fcLayer(x, inputD, outputD, reluFlag, name):
    """fully-connect"""
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape = [inputD, outputD], dtype = "float")
        b = tf.get_variable("b", [outputD], dtype = "float")
        out = tf.nn.xw_plus_b(x, w, b, name = scope.name)
        if reluFlag:
            return tf.nn.relu(out)
        else:
            return out

def convLayer(x, kHeight, kWidth, strideX, strideY,
              featureNum, name, padding = "SAME"):
    """convlutional"""
    channel = int(x.get_shape()[-1])
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape = [kHeight, kWidth, channel, featureNum])
        b = tf.get_variable("b", shape = [featureNum])
        featureMap = tf.nn.conv2d(x, w, strides = [1, strideY, strideX, 1], padding = padding)
        out = tf.nn.bias_add(featureMap, b)
        return tf.nn.relu(tf.reshape(out, featureMap.get_shape().as_list()), name = scope.name)

class VGG19(object):
    """VGG model"""
    def __init__(self, x, keepPro, classNum, skip, modelPath = "vgg19.npy"):
        self.X = x
        self.KEEPPRO = keepPro
        self.CLASSNUM = classNum
        self.SKIP = skip
        self.MODELPATH = modelPath
        #build CNN
        self.buildCNN()

    def buildCNN(self):
        """build model"""
        conv1_1 = convLayer(self.X, 3, 3, 1, 1, 64, "conv1_1" )
        conv1_2 = convLayer(conv1_1, 3, 3, 1, 1, 64, "conv1_2")
        pool1 = maxPoolLayer(conv1_2, 2, 2, 2, 2, "pool1")

        conv2_1 = convLayer(pool1, 3, 3, 1, 1, 128, "conv2_1")
        conv2_2 = convLayer(conv2_1, 3, 3, 1, 1, 128, "conv2_2")
        pool2 = maxPoolLayer(conv2_2, 2, 2, 2, 2, "pool2")

        conv3_1 = convLayer(pool2, 3, 3, 1, 1, 256, "conv3_1")
        conv3_2 = convLayer(conv3_1, 3, 3, 1, 1, 256, "conv3_2")
        conv3_3 = convLayer(conv3_2, 3, 3, 1, 1, 256, "conv3_3")
        conv3_4 = convLayer(conv3_3, 3, 3, 1, 1, 256, "conv3_4")
        pool3 = maxPoolLayer(conv3_4, 2, 2, 2, 2, "pool3")

        conv4_1 = convLayer(pool3, 3, 3, 1, 1, 512, "conv4_1")
        conv4_2 = convLayer(conv4_1, 3, 3, 1, 1, 512, "conv4_2")
        conv4_3 = convLayer(conv4_2, 3, 3, 1, 1, 512, "conv4_3")
        conv4_4 = convLayer(conv4_3, 3, 3, 1, 1, 512, "conv4_4")
        pool4 = maxPoolLayer(conv4_4, 2, 2, 2, 2, "pool4")

        conv5_1 = convLayer(pool4, 3, 3, 1, 1, 512, "conv5_1")
        conv5_2 = convLayer(conv5_1, 3, 3, 1, 1, 512, "conv5_2")
        conv5_3 = convLayer(conv5_2, 3, 3, 1, 1, 512, "conv5_3")
        conv5_4 = convLayer(conv5_3, 3, 3, 1, 1, 512, "conv5_4")
        pool5 = maxPoolLayer(conv5_4, 2, 2, 2, 2, "pool5")

        fcIn = tf.reshape(pool5, [-1, 7*7*512])
        fc6 = fcLayer(fcIn, 7*7*512, 4096, True, "fc6")
        dropout1 = dropout(fc6, self.KEEPPRO)

        fc7 = fcLayer(dropout1, 4096, 4096, True, "fc7")
        dropout2 = dropout(fc7, self.KEEPPRO)

        self.fc8 = fcLayer(dropout2, 4096, self.CLASSNUM, True, "fc8")

    def loadModel(self, sess):
        """load model"""
        wDict = np.load(self.MODELPATH, encoding = "bytes").item()
        #for layers in model
        for name in wDict:
            if name not in self.SKIP:
                with tf.variable_scope(name, reuse = True):
                    for p in wDict[name]:
                        if len(p.shape) == 1:
                            #bias
                            sess.run(tf.get_variable('b', trainable = False).assign(p))
                        else:
                            #weights
                            sess.run(tf.get_variable('w', trainable = False).assign(p))
