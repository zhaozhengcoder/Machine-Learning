import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
# 如果/tmp/data 下面没有这个minist，就会在这个路径下面下载一个 （如果网络不好，最好把shadowscok设置成全局翻墙，这样下载不会报错）
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

n_input=784
n_hidden_1=256
n_hidden_2=128

# 其他参数
batch_size = 256
examples_to_show = 10
display_step = 1
examples_to_show = 10

# 定义输入
input_x= tf.placeholder("float",[None,n_input])

# 编码层的权重
encode_layer1_weight=tf.Variable(tf.random_normal([n_input,n_hidden_1]))
encode_layer2_weight=tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2]))
encode_layer1_bais=tf.Variable(tf.random_normal([n_hidden_1]))
encode_layer2_bais=tf.Variable(tf.random_normal([n_hidden_2]))

# 解码层的权重
decode_layer1_weight=tf.Variable(tf.random_normal([n_hidden_2,n_hidden_1]))
decode_layer2_weight=tf.Variable(tf.random_normal([n_hidden_1,n_input]))
decode_layer1_bais=tf.Variable(tf.random_normal([n_hidden_1]))
decode_layer2_bais=tf.Variable(tf.random_normal([n_input]))

def encode(input):
    h1_output=tf.nn.sigmoid(tf.add(tf.matmul(  input  ,encode_layer1_weight),encode_layer1_bais)) # [batch,784] *[784*256]
    h2_output=tf.nn.sigmoid(tf.add(tf.matmul(h1_output,encode_layer2_weight),encode_layer2_bais)) # [batch,256] * [256,128]
    return h2_output


def decode(input):
    h1_output=tf.nn.sigmoid(tf.add(tf.matmul(input,decode_layer1_weight),decode_layer1_bais))    #[batch, 128] * [128,256]
    h2_output=tf.nn.sigmoid(tf.add( tf.matmul(h1_output,decode_layer2_weight),decode_layer2_bais))  # [batch,256] * [256,784]
    return h2_output


#
encode_res=encode(input_x)
decode_res=decode(encode_res)

y_pred=decode_res
y_true=input_x

#loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_true - y_pred)))
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
# 这里使用的是adam 优化，刚开始的时候，使用了梯度下降，无法收敛
train_step = tf.train.AdamOptimizer(0.01).minimize(loss)


# 开始计算
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

total_batch = int(mnist.train.num_examples/batch_size)
print ("total batch : ",total_batch)

for epoch in range(5):
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        #其实 只需要迭代step就好，but这样就不知道loss了，所以最好写成[train_step,loss]
        #或者写成这样，只不过就不知道loss了 --> # sess.run(train_step , feed_dict={input_x: batch_xs})
        res = sess.run([train_step,loss] , feed_dict={input_x: batch_xs})
    if epoch % display_step == 0:
        print("Epoch: ", epoch+1," cost= ",float(res))

print("Optimization Finished!")


encode_decode = sess.run( y_pred,feed_dict={input_x : mnist.test.images[:examples_to_show]} )
f, a = plt.subplots(2, 10, figsize=(10, 2))
for i in range(examples_to_show):
    a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
plt.show()

sess.close()
