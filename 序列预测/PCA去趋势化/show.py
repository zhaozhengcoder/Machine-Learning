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


def show(main_x,rest_x):
    #for item in main_x:
    #    plt.plot(item)
    for item in rest_x:
        plt.plot(item)
    plt.show()
    

if __name__=="__main__":
    arr = myload("dump_arr.txt")
    main_x = myload("dump_main_x.txt")
    rest_x = myload("dump_rest_x.txt")
    show(arr,arr)


