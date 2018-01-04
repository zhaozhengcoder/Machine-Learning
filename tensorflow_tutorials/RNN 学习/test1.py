import numpy as np
#import tensorflow as tf
#from tensorflow.contrib import rnn
import random
import collections
import time

# Text file containing words for training
training_file = 'belling_the_cat.txt'

def read_data(fname):
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    content = [content[i].split() for i in range(len(content))]
    content = np.array(content)
    #print (content.shape)  #(1, 204)
    content = np.reshape(content, [-1, ])
    #print (content.shape)  #(204,)
    return content

training_data = read_data(training_file)
print ("Loaded training data...")

# 建立字典和反向字典
def build_dataset(words):
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

def mybuild_dataset(words):  # words -- > ['hello','hello','world','python','tensorflow','rnn']
    count = collections.Counter(words)  #Counter({'hello': 2, 'python': 1, 'rnn': 1, 'tensorflow': 1, 'world': 1})
    dictionary=dict()
    for key in count:
        dictionary[key]=len(dictionary)
    #dictionary -- > {'hello': 0, 'python': 3, 'rnn': 1, 'tensorflow': 2, 'world': 4}
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    #reverse_dictionary -- > {0: 'hello', 1: 'rnn', 2: 'tensorflow', 3: 'python', 4: 'world'}
    return dictionary, reverse_dictionary

    


mytrain=['hello','hello','world','python','tensorflow','rnn']
dictionary, reverse_dictionary = mybuild_dataset(mytrain)

print ("dictionary : ",dictionary)
print ("reverse_dictionary : " , reverse_dictionary)

vocab_size = len(dictionary)