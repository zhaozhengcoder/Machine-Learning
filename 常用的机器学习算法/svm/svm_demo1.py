"""
一个　线性可分svm分类器　的demo
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
#style.use("ggplot")
from sklearn import svm

"""
# 这是原始的数据
x = [1, 5, 1.5, 8, 1, 9]
y = [2, 8, 1.8, 8, 0.6, 11]

plt.scatter(x,y)
plt.show()
"""

# 转换成　样本数　×　特征　的格式
X = np.array([[1,2],[5,8],[1.5,1.8],[8,8],[1,0.6],[9,11]])
# y 表示的是label
y = [0,1,0,1,0,1]


# 惩罚因子C取1.0。如果你不知道C的作用也不用着急，姑且看成是对分类器表现的调节参数
# sklearn 文档　，https://xacecask2.gitbooks.io/scikit-learn-user-guide-chinese-version/content/sec1.4.html 
clf = svm.SVC(kernel='linear', C = 1.0)
clf.fit(X,y)

# clf.coef_ 可以理解为　weights * x + bias ,clf.coef_  只存在与线性核里面
# 这是一个二维平面的分类，所以分类直线的方程是　theta0 * x1 + theta1 * x2 + bias = 0
# clf.coef_[0] 里面的两个数，分别是　theta0 和　theta1 ,intercept_[0] 指的是bias 
w = clf.coef_[0]
print(w)

xx = np.linspace(0,12)
# a 是斜率
a = -w[0] / w[1]
# clf.intercept_[0] / w[1] 是截距
yy = a * xx - clf.intercept_[0] / w[1]

h0 = plt.plot(xx, yy, 'k-', label="non weighted div")

plt.scatter(X[:, 0], X[:, 1], c = y)

pred_x1=0.58
pred_y1=0.76

pred_x2=10.58
pred_y2=10.76

# 预测一个点的类别
print ("x1 and y1 的类别是　：",clf.predict(np.array( [[pred_x1,pred_y1]] )))
print ("x2 and y2 的类别是　：",clf.predict(np.array( [[pred_x2,pred_y2]] )))

plt.scatter(0.58, 0.76,color="red")
plt.scatter(10.58,10.76,color="red")

plt.legend()
plt.show()


