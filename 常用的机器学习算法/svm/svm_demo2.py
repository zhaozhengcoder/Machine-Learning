"""
一个　非线性svm分类器　的demo
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import svm
#style.use("ggplot")


# 构造了一个非线性的数据
X = np.array([[1,1],[5,5],[5,1],[1,5],[2.5,2.5],[2,3],[3,2],[2,2],[3,3]])
y = [1,1,1,1,0,0,0,0,0]
plt.scatter(X[:, 0], X[:, 1], c = y)
#plt.show()

#svc = svm.SVC(kernel='linear',C=1.0) # 线性核
#svc = svm.SVC(kernel='rbf',C=1.0,gamma=1)  #如果是非线性核　可以还不同的gamma值
svc = svm.SVC(kernel='rbf',C=1.0,gamma='auto')
svc.fit(X,y)

h = 0.01
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))


plt.subplot(1, 1, 1)
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlim(xx.min(), xx.max())
plt.title("SVC with kernel "+svc.kernel)
plt.show()
