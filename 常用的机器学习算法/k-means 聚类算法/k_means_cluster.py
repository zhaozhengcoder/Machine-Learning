# -*- coding: utf-8 -*

from matplotlib import pyplot as plt
from sklearn.cluster import k_means
import pandas as pd
from sklearn.metrics import silhouette_score

file = pd.read_csv("cluster_data.csv", header=0)
X = file['x']
y = file['y']


# 看一下不同的k值的效果
def k_number():
    index = []
    inertia = []
    silhouette = []
    for i in range(20):
        model = k_means(file, n_clusters=i + 2)
        # model[2]
        # 准确说来只是一个数值，它代表着所有样本点距离最近中心点距离的总和。
        # 你可以 在大脑里想一下，当我们的 K 值增加时，也就是类别增加时，这个数值应该是会降低的。 
        inertia.append(model[2])
        index.append(i + 2)
        # 轮廓系数
        # 轮廓系数综合了聚类后的两项 因素：内聚度和分离度。
        # 内聚度就指一个样本在簇内的不相似度，而分离度就指一个样本 在簇间的不相似度。
        silhouette.append(silhouette_score(file, model[1]))
    print (silhouette) 
    plt.plot(index, silhouette, "-o")
    plt.plot(index, inertia, "-o")
    plt.show()


def k_means(n_cluster):
    model = k_means(file, n_clusters=n_cluster)
    cluster_centers = model[0]
    cluster_labels = model[1]

    cluster_inertia = model[2]
    plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    ax1.scatter(X, y, c="grey")
    ax2.scatter(X, y, c=cluster_labels)

    for center in cluster_centers:
        ax2.scatter(center[0], center[1], marker="p", edgecolors="red")
    print ("cluster_inertia: %s" % cluster_inertia)

    plt.show()


if __name__ == '__main__':
    #plt.scatter(X,y)
    #plt.show()
    #k_number()
    k_means(int(input("Input clusters: ")))