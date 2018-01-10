import numpy as np 


modelPath = "vgg19.npy"
wDict = np.load(modelPath, encoding = "bytes").item()
for name in wDict:
    print (name)
    """
    conv3_1
    conv1_2
    conv4_1
    conv4_3
    conv1_1
    conv5_1
    conv3_3
    conv4_2
    fc7
    conv2_2
    conv5_2
    conv5_3
    conv2_1
    conv5_4
    conv4_4
    fc6
    fc8
    conv3_4
    conv3_2
    """

"""
# 全连接层的维度
c['fc6'] is list ,length is 2 , [0] is weights ,[1] is bias 

In [10]: c['fc6'][0].shape
Out[10]: (25088, 4096)

In [11]: c['fc6'][1].shape
Out[11]: (4096,)


# 卷积层的维度
In [5]: len(wDict['conv1_1'])
Out[5]: 2

In [6]: wDict['conv1_1'][0].shape
Out[6]: (3, 3, 3, 64)

In [7]: wDict['conv1_1'][1].shape
Out[7]: (64,)

"""



"""
for name in wDict:
    for index,p in enumerate(wDict[name]):
        print (name," - [",index,"]  ",p.shape)
"""
"""

fc7  - [ 0 ]   (4096, 4096)
fc7  - [ 1 ]   (4096,)
conv3_3  - [ 0 ]   (3, 3, 256, 256)
conv3_3  - [ 1 ]   (256,)
conv5_3  - [ 0 ]   (3, 3, 512, 512)
conv5_3  - [ 1 ]   (512,)
conv5_1  - [ 0 ]   (3, 3, 512, 512)
conv5_1  - [ 1 ]   (512,)
conv5_2  - [ 0 ]   (3, 3, 512, 512)
conv5_2  - [ 1 ]   (512,)
conv3_2  - [ 0 ]   (3, 3, 256, 256)
conv3_2  - [ 1 ]   (256,)
conv4_4  - [ 0 ]   (3, 3, 512, 512)
conv4_4  - [ 1 ]   (512,)
fc8  - [ 0 ]   (4096, 1000)
fc8  - [ 1 ]   (1000,)
conv3_4  - [ 0 ]   (3, 3, 256, 256)
conv3_4  - [ 1 ]   (256,)
conv5_4  - [ 0 ]   (3, 3, 512, 512)
conv5_4  - [ 1 ]   (512,)
conv2_1  - [ 0 ]   (3, 3, 64, 128)
conv2_1  - [ 1 ]   (128,)
conv1_1  - [ 0 ]   (3, 3, 3, 64)
conv1_1  - [ 1 ]   (64,)
conv4_3  - [ 0 ]   (3, 3, 512, 512)
conv4_3  - [ 1 ]   (512,)
conv4_1  - [ 0 ]   (3, 3, 256, 512)
conv4_1  - [ 1 ]   (512,)
conv1_2  - [ 0 ]   (3, 3, 64, 64)
conv1_2  - [ 1 ]   (64,)
conv2_2  - [ 0 ]   (3, 3, 128, 128)
conv2_2  - [ 1 ]   (128,)
conv4_2  - [ 0 ]   (3, 3, 512, 512)
conv4_2  - [ 1 ]   (512,)
conv3_1  - [ 0 ]   (3, 3, 128, 256)
conv3_1  - [ 1 ]   (256,)
fc6  - [ 0 ]   (25088, 4096)
fc6  - [ 1 ]   (4096,)
"""


"""
for name in wDict:
    for index,p in enumerate(wDict[name]):
        print (name," - [",index,"]  ",len(p.shape))


fc7  - [ 0 ]   2
fc7  - [ 1 ]   1
conv3_3  - [ 0 ]   4
conv3_3  - [ 1 ]   1
conv5_3  - [ 0 ]   4
conv5_3  - [ 1 ]   1
conv5_1  - [ 0 ]   4
conv5_1  - [ 1 ]   1
conv5_2  - [ 0 ]   4
conv5_2  - [ 1 ]   1
conv3_2  - [ 0 ]   4
conv3_2  - [ 1 ]   1
conv4_4  - [ 0 ]   4
conv4_4  - [ 1 ]   1
fc8  - [ 0 ]   2
fc8  - [ 1 ]   1
conv3_4  - [ 0 ]   4
conv3_4  - [ 1 ]   1
conv5_4  - [ 0 ]   4
conv5_4  - [ 1 ]   1
conv2_1  - [ 0 ]   4
conv2_1  - [ 1 ]   1
conv1_1  - [ 0 ]   4
conv1_1  - [ 1 ]   1
conv4_3  - [ 0 ]   4
conv4_3  - [ 1 ]   1
conv4_1  - [ 0 ]   4
conv4_1  - [ 1 ]   1
conv1_2  - [ 0 ]   4
conv1_2  - [ 1 ]   1
conv2_2  - [ 0 ]   4
conv2_2  - [ 1 ]   1
conv4_2  - [ 0 ]   4
conv4_2  - [ 1 ]   1
conv3_1  - [ 0 ]   4
conv3_1  - [ 1 ]   1
fc6  - [ 0 ]   2
fc6  - [ 1 ]   1

"""