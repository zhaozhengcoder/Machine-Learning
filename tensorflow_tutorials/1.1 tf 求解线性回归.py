# tensorflow hello world
# run in python 3.5 

import tensorflow as tf
import numpy as np

# create data
# 创建一个随机的数据集
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3

# 随机初始化 权重
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

# 估计的y值
y = Weights*x_data + biases

# 估计的y和真实的y，计算cost
loss = tf.reduce_mean(tf.square(y-y_data))

# 梯度下降优化
optimizer = tf.train.GradientDescentOptimizer(0.5)  # 0.5 学习率
train = optimizer.minimize(loss)

"""
到目前为止, 我们只是建立了神经网络的结构, 还没有使用这个结构. 
在使用这个结构之前, 我们必须先初始化所有之前定义的Variable, 所以这一步是很重要的
"""
# init = tf.initialize_all_variables() # tf 马上就要废弃这种写法
init = tf.global_variables_initializer()  # 替换成这样就好



#创建会话
sess = tf.Session()
sess.run(init)          #  用 Session来 run 每一次 training 的数据.


print (x_data)
print (y_data)

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))

"""
输出：
0 [ 0.08055115] [ 0.42646995]
20 [ 0.07926457] [ 0.31104472]
40 [ 0.09364213] [ 0.30338651]
60 [ 0.09805057] [ 0.30103838]
80 [ 0.09940226] [ 0.30031839]
100 [ 0.09981672] [ 0.30009764]
120 [ 0.0999438] [ 0.30002993]
140 [ 0.09998276] [ 0.30000919]
160 [ 0.0999947] [ 0.30000284]
180 [ 0.09999838] [ 0.30000088]
200 [ 0.09999951] [ 0.30000028]

这个结果基本上的对的，我们在构造x-data 和y-data的时候，就是y-data=x-data * 0.3 + 1 
也就是说 w=0.3 b=1 


如果想了解公司背后的意义： 

线性回归： http://www.jianshu.com/p/ecb36d1441af

梯度下降的公司推导 ： http://www.jianshu.com/p/200591639c2c
"""