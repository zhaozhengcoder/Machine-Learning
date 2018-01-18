# 这是一个ｐｃａ的ｄｅｍｏ
# 这是吴恩达 教程上面的那个例子 
# 写在打印出来的了讲义上面了

from sklearn import datasets
from sklearn import decomposition
import numpy as np


arr = np.array([[-1,-1,0,2,0],[-2,0,0,1,1]])
X = arr.T   


# n_components=2 保留的主成分特征的数量
# 建立pca 模型
estimator = decomposition.PCA(n_components=1)

# 进行 pca 处理
# reduce_data.shape ：(1797, 2)
# X.shape : (1797, 64)
reduce_data = estimator.fit_transform(X)


print ("降维前, shape : ",X.shape)
print (X)

print ("降维后, shape : ",reduce_data.shape)
print (reduce_data)