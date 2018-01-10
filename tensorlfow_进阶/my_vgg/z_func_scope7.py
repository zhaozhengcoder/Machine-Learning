"""
一个基本的模型，可以看懂一下，后面就比较好理解了
"""

import tensorflow as tf 
import numpy as np 
import caffe_classes
import cv2



def maxPoolLayer(x, kHeight, kWidth, strideX, strideY, name, padding = "SAME"):
    """max-pooling"""
    return tf.nn.max_pool(x, ksize = [1, kHeight, kWidth, 1],strides = [1, strideX, strideY, 1], padding = padding, name = name)

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

def convLayer(x, kHeight, kWidth, strideX, strideY,featureNum, name, padding = "SAME"):
    """convlutional"""
    channel = int(x.get_shape()[-1])
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape = [kHeight, kWidth, channel, featureNum])
        b = tf.get_variable("b", shape = [featureNum])
        featureMap = tf.nn.conv2d(x, w, strides = [1, strideY, strideX, 1], padding = padding)
        out = tf.nn.bias_add(featureMap, b)
        return tf.nn.relu(tf.reshape(out, featureMap.get_shape().as_list()), name = scope.name)

def buildCNN(input_x):
    # 基本的参数
    KEEPPRO = 1
    CLASSNUM = 1000

    # 定义网络的结构 VGG的 网络结构 5个卷积层 和3个全连接层
    conv1_1 = convLayer(input_x, 3, 3, 1, 1, 64, "conv1_1" )
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
    dropout1 = dropout(fc6, KEEPPRO)

    fc7 = fcLayer(dropout1, 4096, 4096, True, "fc7")
    dropout2 = dropout(fc7, KEEPPRO)

    fc8 = fcLayer(dropout2, 4096, CLASSNUM, True, "fc8")
    return fc8  # 网络模型的输出

def loadModel():
    """load model"""
    # 加载预训练数据 
    #for layers in model
    print ("load --------")
    modelPath = "vgg19.npy"
    wDict = np.load(modelPath,encoding = "bytes").item()

    for name in wDict:
        with tf.variable_scope(name ,reuse =True):
            for p in wDict[name]:
                if len(p.shape) == 1:
                    #bias
                    #这个sess.run 很重要，没有的话，程序虽然可以运行，but 没有办法有效的分类图片
                    sess.run(tf.get_variable('b', trainable = False).assign(p))
                else:
                    #weights
                    #这个sess.run 很重要，没有的话，程序虽然可以运行，but 没有办法有效的分类图片
                    sess.run(tf.get_variable('w', trainable = False).assign(p))




def pre_input():
    path='demo1.png'
    cv2.imread(path)
    img=cv2.imread(path)
    imgMean = np.array([104, 117, 124], np.float32)
    resized = cv2.resize(img.astype(np.float32), (224, 224)) - imgMean
    # 将一张图片转化为 tf 需要的格式的矩阵
    x_input = resized.reshape((1,224,224,3))
    return x_input


x_input = pre_input()
x = tf.placeholder("float", [1, 224, 224, 3])



# 放进vgg模型里面
out = buildCNN(x)
# 讲输出结果softmax 找到对应的标签
softmax = tf.nn.softmax(out)


# 
init = tf.global_variables_initializer() 
with tf.Session() as sess:
    sess.run(init)
    loadModel()
    
    #print (sess.run(x,feed_dict = {x: x_input}))
    print (sess.run(x,feed_dict = {x: x_input}).shape )
    
    maxx = np.argmax(sess.run(softmax, feed_dict = {x: x_input}))
    res = caffe_classes.class_names[maxx]
    print (res)

    

    


