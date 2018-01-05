import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

"""
这是一个7层的网络
(sample * 1)-->(1,30)-->(30,30)-->(30,30)-->(30,30)-->(30,30)-->(30,30)-->(30,1)

这已经是一个比较深的神经网络了
如果不使用批标准化，那么传统的神经网络就会出现梯度消失

但是，使用了批标准化之后，就不会出现这种情况


这个文件增加了 plot_his 函数
"""


ACTIVATION = tf.nn.sigmoid
#ACTIVATION = tf.nn.relu
N_LAYERS = 7
N_HIDDEN_UNITS = 30

def fix_seed(seed=1):
    # reproducible
    np.random.seed(seed)
    tf.set_random_seed(seed)

def plot_his(inputs, inputs_norm):
    # plot histogram for the inputs of every layer
    for j, all_inputs in enumerate([inputs, inputs_norm]):
        for i, input in enumerate(all_inputs):
            plt.subplot(2, len(all_inputs), j*len(all_inputs)+(i+1))
            plt.cla()
            if i == 0:
                the_range = (-7, 10)
            else:
                the_range = (-1, 1)
            plt.hist(input.ravel(), bins=15, range=the_range, color='#FF5733')
            plt.yticks(())
            if j == 1:
                plt.xticks(the_range)
            else:
                plt.xticks(())
            ax = plt.gca()
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
        plt.title("%s normalizing" % ("Without" if j == 0 else "With"))
    plt.draw()
    plt.pause(0.01)



def add_layer(inputs,in_size,out_size,activation_function=None,norm=False):
    Weights = tf.Variable(tf.random_normal([in_size,out_size],mean=0., stddev=1.))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    
    if norm == False:  # 不使用批标准化，也就是正常的情况
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs
    else:   # 使用批标准化
        fc_mean, fc_var = tf.nn.moments(Wx_plus_b,axes=[0],)
        # scale 是扩大倍数 , shift是平移
        scale = tf.Variable(tf.ones([out_size]))
        shift = tf.Variable(tf.zeros([out_size]))
        epsilon = 0.001
        Wx_plus_b = tf.nn.batch_normalization(Wx_plus_b, fc_mean, fc_var, shift, scale, epsilon)

        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs


"""
关于 tf.nn.moments  ，就是计算 均值和方差 
img = tf.Variable(tf.random_normal([2, 3]))
mean, variance = tf.nn.moments(img,[0])

输出 ：
img  =     [[ 0.69495416  2.08983064 -1.08764684]
            [ 0.31431156 -0.98923939 -0.34656194]]
mean =      [ 0.50463283  0.55029559 -0.71710438]
variance =  [ 0.0362222   2.37016821  0.13730171]
"""



def build_net(xs,ys,norm):
    #如果使用批处理
    if norm:
        #对输入进行批处理   
        fc_mean, fc_var = tf.nn.moments(xs,axes=[0],)
        scale = tf.Variable(tf.ones([1]))
        shift = tf.Variable(tf.zeros([1]))
        epsilon = 0.001 
        xs = tf.nn.batch_normalization(xs, fc_mean, fc_var, shift, scale, epsilon)
    
    # 没有批处理的从这里开始执行
    # 记录每一层的输入
    layers_inputs=[xs]

    for l_n in range(N_LAYERS):
        layer_input = layers_inputs[l_n]                     # 拿出上一层的输入
        in_size = layers_inputs[l_n].get_shape()[1].value    # 输出的维度
        
        #show 
        #print ("no : ",l_n , " : ", layers_inputs[l_n].get_shape()," , get_shape()[1] : ",layers_inputs[l_n].get_shape()[1])
        print ("in size : ",in_size)
        output= add_layer(layer_input,in_size,N_HIDDEN_UNITS,ACTIVATION,norm)
        #这一层的输出，是下一层的输入
        layers_inputs.append(output)
    
    # 最后的预测值
    prediction = add_layer(layers_inputs[-1], 30, 1, activation_function=None)
    # 计算cost
    cost = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
    # layers_inputs 每一层的输入（或者可以理解为下一层的输出）
    # 所以这里返回的是一个list
    return [train_op, cost, layers_inputs]



# make up data
fix_seed(1)
x_data = np.linspace(-7, 10, 2500)[:, np.newaxis]
np.random.shuffle(x_data)
noise = np.random.normal(0, 8, x_data.shape)
y_data = np.square(x_data) - 5 + noise

# plot input data
#plt.scatter(x_data, y_data)
#plt.show()

xs = tf.placeholder(tf.float32, [None, 1])  # [num_samples, num_features]
ys = tf.placeholder(tf.float32, [None, 1])

train_op, cost, layers_inputs = build_net(xs, ys, norm=False)   # without BN
train_op_norm, cost_norm, layers_inputs_norm = build_net(xs, ys, norm=True) # with BN


sess = tf.Session()
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

# record cost
cost_his = []
cost_his_norm = []
record_step = 5

#plt.ion()  # 开启交互模型 ，图像会一闪而过
#plt.figure(figsize=(7, 3))  # 指定大小
for i in range(250):
    if i % 50 == 0:
        # plot histogram
        # all_inputs 和 all_inputs_norm 分别表示的是，有批处理和无批处理两种情况下，分别的每一层的输入值（8个输入），这一个list的type
        all_inputs, all_inputs_norm = sess.run([layers_inputs, layers_inputs_norm], feed_dict={xs: x_data, ys: y_data})
        print ("all_inputs : ",len(all_inputs))
        print ("all_inputs_norm : ",len(all_inputs_norm))
        #print ("  --> : " ,all_inputs[0])                            #len -->  8
        print ("all_inputs[0] , shape --> : " ,all_inputs[0].shape)   #all_inputs[0] , shape --> :  (2500, 1)
        print ("all_inputs[1] , shape --> : " ,all_inputs[1].shape)   #all_inputs[1] , shape --> :  (2500, 30)
        print ("all_inputs[2] , shape --> : " ,all_inputs[2].shape)   #all_inputs[2] , shape --> :  (2500, 30)
        print ("all_inputs[3] , shape --> : " ,all_inputs[3].shape)   #all_inputs[3] , shape --> :  (2500, 30)
        print ("all_inputs[7] , shape --> : " ,all_inputs[7].shape)   #all_inputs[7] , shape --> :  (2500, 30)
        #plot_his(all_inputs, all_inputs_norm)

    # train on batch
    sess.run([train_op, train_op_norm], feed_dict={xs: x_data[i*10:i*10+10], ys: y_data[i*10:i*10+10]})

    if i % record_step == 0:
        # record cost
        cost_his.append(sess.run(cost, feed_dict={xs: x_data, ys: y_data}))
        cost_his_norm.append(sess.run(cost_norm, feed_dict={xs: x_data, ys: y_data}))

#plt.ioff()

plt.figure()
plt.plot(np.arange(len(cost_his))*record_step, np.array(cost_his), label='no BN')     # no norm
plt.plot(np.arange(len(cost_his))*record_step, np.array(cost_his_norm), label='BN')   # norm
plt.legend()
plt.show()