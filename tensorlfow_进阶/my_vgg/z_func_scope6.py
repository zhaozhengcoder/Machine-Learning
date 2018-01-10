"""
搞定输入，读取一张图片
"""

import os
import urllib.request
import argparse
import sys
import vgg19
import cv2
import tensorflow as tf
import numpy as np
import caffe_classes



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


######## ----- 
dropoutPro = 1
classNum = 1000
skip = []

model = vgg19.VGG19(x, dropoutPro, classNum, skip)
score = model.fc8
softmax = tf.nn.softmax(score)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print (sess.run(x,feed_dict = {x: x_input}))
    print (sess.run(x,feed_dict = {x: x_input}).shape )

    #model.loadModel(sess)
    #maxx = np.argmax(sess.run(softmax, feed_dict = {x: x_input}))
    #res = caffe_classes.class_names[maxx]
    #print (res)


