import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

# mnist 
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

n_input=784
n_layer1=10
examples_to_show=10

# 输入
x_input=tf.placeholder("float",[None,n_input])    # [batch,784]
y_input=tf.placeholder("float",[None,n_layer1])   #

# layer1
layer1_weights=tf.Variable(tf.random_normal([n_input, n_layer1])) #[784,10]
layer1_bais=tf.Variable(tf.random_normal([n_layer1]))

def addlayer(x_input,layer1_weights,layer1_bais,activation_function=None):
    output=tf.add(tf.matmul(x_input,layer1_weights),layer1_bais)
    if activation_function==None:
        return tf.nn.sigmoid(output)
    else:
        #return activation_function(output)
        return tf.nn.softmax(output)


#预测输出
y_pre=addlayer(x_input,layer1_weights,layer1_bais,activation_function=tf.nn.softmax)
y_true=y_input

#反向
cross_entropy=-tf.reduce_sum(y_true * tf.log(y_pre))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy) 


# 计算
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs,batch_ys =mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x_input:batch_xs , y_input:batch_ys})

    if (i%50==0):
        #print ("loss : ",sess.run(cross_entropy))  --> 之前这么写得，都是报错，因为没有加feed_dict 
        print ("loss : ",sess.run(cross_entropy,feed_dict={x_input:batch_xs , y_input:batch_ys}))


#从测试集里面拿一些数据 试一试结果
res=sess.run(y_pre,feed_dict={x_input : mnist.test.images[:examples_to_show]} )
#print (res)
#print (res.shape)  # shape : examples_to_show * 10 

# 输出图片识别的结果
print ("number is : ", sess.run(tf.argmax(res,1)))

# show 一下图片
# 这段代码 是从别的地方copy 过来的，作用是将图片show出来
f, a = plt.subplots(2, 10, figsize=(10, 2))
for i in range(examples_to_show):
    a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    #a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
plt.show()

sess.close()