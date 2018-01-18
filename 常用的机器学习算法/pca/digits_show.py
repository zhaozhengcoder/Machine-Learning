# 这个是sklearn带的一个手写字符集合
# 可以熟悉一下这个数据集合，然后使用pca + kmeans 对这个数据进行一个处理

from sklearn import datasets
from matplotlib import pyplot as plt
from sklearn import decomposition
from sklearn.cluster import KMeans
import numpy as np

digits_data = datasets.load_digits()

for index,image in enumerate(digits_data.images[:5]):
    plt.subplot(2,5,index+1)
    plt.imshow(image)

plt.show()