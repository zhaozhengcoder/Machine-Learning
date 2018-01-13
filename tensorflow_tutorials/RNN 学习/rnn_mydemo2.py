import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import collections
import time

## 预处理数据
def read_file():
    # Text file containing words for training
    training_file = 'belling_the_cat.txt'
    content=[]
    with open(training_file,'r') as f:
        for line in f.readlines():
            # line 表示读到数据的每一行，linelist是按照空格切分成一个list
            linelist=line.strip().split()
            for i in linelist:
                content.append(i.strip())
    content=np.array(content)
    content=np.reshape(content,[-1,])  #shape (204,1)
    return content

def mybuild_dataset(words):  
    # words -- > ['hello','hello','world','python','tensorflow','rnn']
    count = collections.Counter(words)  
    # Counter({'hello': 2, 'python': 1, 'rnn': 1, 'tensorflow': 1, 'world': 1})
    dictionary=dict()
    for key in count:
        dictionary[key]=len(dictionary)
    #dictionary -- > {'hello': 0, 'python': 3, 'rnn': 1, 'tensorflow': 2, 'world': 4}
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    #reverse_dictionary -- > {0: 'hello', 1: 'rnn', 2: 'tensorflow', 3: 'python', 4: 'world'}
    return dictionary, reverse_dictionary  #len(dictionary)  --> 112  

training_data=read_file()
dictionary, reverse_dictionary=mybuild_dataset(training_data)
vocab_size = len(dictionary)   # size : 112 

## RNN 模型
# Parameters
learning_rate = 0.001
training_iters = 5000
display_step = 1000
n_input = 3

# number of units in RNN cell
n_hidden = 512

# tf Graph input
x = tf.placeholder("float", [None, n_input, 1])
y = tf.placeholder("float", [None, vocab_size])   #size: 112 

# RNN output node weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]))  # n_hidden =512 , vocab_size=112
}
biases = {
    'out': tf.Variable(tf.random_normal([vocab_size]))
}

def RNN(x, weights, biases):
    batch_size=1
    x = tf.reshape(x, [batch_size,n_input,1])          # (1,3,1) 相当于batch =1 
    # rnn 
    cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    # final_state 的维度是  batch * n_hidden                       --> 1 * 512
    # outputs     的维度是  batch * n_input(time_step) * n_hidden  --> 1 * 3  * 512
    outputs, final_state = tf.nn.dynamic_rnn(cell, x, initial_state=init_state, time_major=False)  
    
    #print ("before unstack , output shape : ",outputs.shape)   # output shape :  (1,3,512) (batch,time_step,cell_n_hidden)
    #unstack 更改维度
    outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
    #这个时候 outputs 变成了list 
    #print ("output shape[-1] 2: ",outputs[-1].shape)           # output shape :  (3,1,512), outputs[-1] shape (1,512)
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']
    #(1,112)  这个的表示意义是一个(1,112)的onehot，112表示字典里面总共有112个词汇
    return results   #(1, 112)  这个表示的是一个onehot

pred = RNN(x, weights, biases)
# Loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

# Model evaluation
# tf.argmax(pred,1) 返回一个 pred (1,112) 的每一行的最大值，这个返回值是一个数，表示这一行的最大的下标
# tf.argmax(y,1)    返回也是一个数，表示这一行最大的下标 
# 如果 equal(A,B)  A=[1] ,B=[0] ，输出 => [False] ,如果 A=[1] ,B=[1] ，输出 => [true]
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
# 然后将bool 转换成 int ，来进行求和
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as session:
    session.run(init)
    step = 0
    #从 0~4 随机选一个作为起点,每次随机找一个起点，然后切分数据，取前三个是x，取第四个是y
    offset = random.randint(0,n_input+1)
    end_offset = n_input + 1
    acc_total = 0
    loss_total = 0

    while step < training_iters:
        if offset > (len(training_data)-end_offset):
            offset = random.randint(0, n_input+1)

        # 输入x ，将前三个词汇转换成词向量
        # symbols_in_keys  是一个二维的list -->  [[34], [92], [85]]
        symbols_in_keys = [[dictionary[ str(training_data[i])]] for i in range(offset, offset+n_input) ]
        # reshape  把他们转换成 (1, 3, 1)
        symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])

        # 这一段代码搞定是 y_true ，把第四个词转换成词向量 onehot的类型
        symbols_out_onehot = np.zeros([vocab_size], dtype=float)
        # str(training_data[offset+n_input])  ->  'mice'
        symbols_out_onehot[dictionary[str(training_data[offset+n_input])]] = 1.0
        symbols_out_onehot = np.reshape(symbols_out_onehot,[1,-1])

        # 完成一次训练
        #res_pred=session.run(pred,feed_dict={x: symbols_in_keys, y: symbols_out_onehot})
        #print ("pred : ",res_pred)
        #print ("pred shape : ",res_pred.shape)  # pred shape :  (1, 112)

        """
        session.run(cost,feed_dict={x: symbols_in_keys, y: symbols_out_onehot})
        if (step %1000==0):
            print (session.run(cost,feed_dict={x: symbols_in_keys, y: symbols_out_onehot}))
        """
        _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred],feed_dict={x: symbols_in_keys, y: symbols_out_onehot})
        if(step % 1000==0):
            #print ("loss : ",loss," , acc : ",acc," , onehot_pred : ",onehot_pred )
            print ("loss : ",loss," , acc : ",acc)

            #测试的输入
            test_x_str1=['a', 'young', 'mouse']
            test_y1='got'
            test_x_str2=['he','thought','would']
            test_y2='meet'

            test_input1=[[dictionary[i]] for i in test_x_str1]
            test_input2=[[dictionary[i]] for i in test_x_str2]
            #reshape 
            test_input1=np.reshape(np.array(test_input1), [-1, n_input, 1])
            test_input2=np.reshape(np.array(test_input2), [-1, n_input, 1])

            test_y1_onehot = np.zeros([vocab_size], dtype=float)
            test_y1_onehot[dictionary[test_y1]] = 1.0
            test_y1_onehot = np.reshape(test_y1_onehot,[1,-1])
            test_y2_onehot = np.zeros([vocab_size], dtype=float)
            test_y2_onehot[dictionary[test_y2]] = 1.0
            test_y2_onehot = np.reshape(test_y2_onehot,[1,-1])

            pred_test1=session.run(pred,feed_dict={x: test_input1 , y: test_y1_onehot})
            pred_test1_word =reverse_dictionary[int(tf.argmax(pred_test1, 1).eval())]

            pred_test2=session.run(pred,feed_dict={x: test_input2 , y: test_y2_onehot})
            pred_test2_word =reverse_dictionary[int(tf.argmax(pred_test2, 1).eval())]

            print (test_x_str1," || the next pred word is : ",pred_test1_word," vs the true word is : ", test_y1)
            print (test_x_str2," || the next pred word is : ",pred_test2_word," vs the true word is : ", test_y2)


        step += 1
        # 这个offset 不要被它开始的随机赋值吓到 其实就是随便找一个点作为起始，然后每次去三个，不断的向后移动
        offset += (n_input+1)