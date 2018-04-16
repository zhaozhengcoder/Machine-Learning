import pandas as pd 
from datetime import datetime 
import os 
import numpy as np 
import matplotlib.pyplot as plt 
import pickle
import pca
import pickle

def myload(filename):
    abs_path='C:/Users/wwwa8/Documents/GitHub/Machine-Learning/序列预测/PCA去趋势化/'
    f = open(abs_path+filename,'rb')
    data =pickle.load(f)
    f.close()
    return data

def show1(arr):
    for item in arr:
        plt.plot(item)
    plt.show()

def show2(main_x,rest_x):
    for item in main_x:
        plt.plot(item)
    for item in rest_x:
        plt.plot(item)
    plt.show()
    
def show_index(arr,index):
    plt.xlabel("Index of time")
    plt.ylabel("Traffic volume")
    plt.plot(arr[index])
    plt.show()
    plt.close()
    
if __name__=="__main__":
    arr = myload("dump_arr_9-13.txt")
    main_x = myload("dump_main_x_9-13.txt")
    rest_x = myload("dump_rest_x_9-13.txt")
    show1(arr)
    show1(main_x)
    show1(rest_x)
    show2(main_x,rest_x)