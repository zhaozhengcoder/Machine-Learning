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

## rnn 
# Parameters
learning_rate = 0.001
training_iters = 1000
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
    # reshape to [1, n_input]
    x = tf.reshape(x, [-1, n_input])
    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    print ("origin x shape : ",x.shape)   # (1,3)

    #现在的x已经变成了list ，原来的shape是 (1,3) ,执行split之后，x是 [array([[1]]), array([[3]]), array([[1]])]
    x = tf.split(x,n_input,1)    

    #print ("last x shape : ",x.shape)

    # 2-layer LSTM, each layer has n_hidden units.
    # Average Accuracy= 95.20% at 50k iter
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden)])
    # 1-layer LSTM with n_hidden units but with lower accuracy.
    # Average Accuracy= 90.60% 50k iter
    # Uncomment line below to test but comment out the 2-layer rnn.MultiRNNCell above
    # rnn_cell = rnn.BasicLSTMCell(n_hidden)
    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
    # there are n_input outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']   # weights['out'] shape : (512,112) , outputs[-1] (1 , 512)* weights['out'] (512,112) == (1 ,112 )

pred = RNN(x, weights, biases)
# Loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as session:
    session.run(init)
    step = 0
    #从 0~4 随机选一个作为起点 
    offset = random.randint(0,n_input+1)
    end_offset = n_input + 1
    acc_total = 0
    loss_total = 0

    symbols_in_keys = [ [dictionary[ str(training_data[i])]] for i in range(offset, offset+n_input) ]
    symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])

    # 这一段代码搞定是 y_true
    symbols_out_onehot = np.zeros([vocab_size], dtype=float)
    #  str(training_data[offset+n_input])  ->  'mice'
    symbols_out_onehot[dictionary[str(training_data[offset+n_input])]] = 1.0
    symbols_out_onehot = np.reshape(symbols_out_onehot,[1,-1])

    # 完成一次训练
    res_pred=session.run(pred,feed_dict={x: symbols_in_keys, y: symbols_out_onehot})
    print ("pred : ",res_pred)
    print ("pred shape : ",res_pred.shape)  # pred shape :  (1, 112)
