"""
使用pca对数据进行降维，然后show出降维之后数据的分布
+
使用降维之后的数据最为kmeans算法的输入，进行聚类，（聚成１０类）
可以观察到，聚类之后数据的分布和降维后加上标签的数据的分布有相似的部分
"""

from sklearn import datasets
from matplotlib import pyplot as plt
from sklearn import decomposition
from sklearn.cluster import k_means
import numpy as np

digits_data = datasets.load_digits()

X = digits_data.data  # shape (1797, 64)  --> 含义 ：（样本 ×　每个样本的特征）
y = digits_data.target  # shape  (1797,)　 --> 含义：每个样本的label,即他们对应的数字

## 使用pca降维
# n_components=2 保留的主成分特征的数量
# 建立pca 模型
estimator = decomposition.PCA(n_components=2)

# 进行 pca 处理
# reduce_data.shape ：(1797, 2)
# X.shape : (1797, 64)
reduce_data = estimator.fit_transform(X)

plt.figure(figsize=(12, 6))
ax1 = plt.subplot(1, 3, 1)
ax2 = plt.subplot(1, 3, 2)
ax3 = plt.subplot(1, 3, 3)

ax1.scatter(reduce_data[:,0],reduce_data[:,1],c=y)


## k-means 聚类
model = k_means(reduce_data,n_clusters=10)
cluster_centers = model[0]
cluster_labels = model[1]
cluster_inertia = model[2]

ax2.scatter(reduce_data[:,0], reduce_data[:,1], c="grey")
ax3.scatter(reduce_data[:,0], reduce_data[:,1], c=cluster_labels)
plt.show()

print ("左图表示降维之后的数据分布，不同颜色代表不同的数字；中图表示降维的数据分布；右图表示对中图的数据进行聚类后的结果