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
    
#给横坐标由原来的0~480，变成现在的0~24，改变横坐标的比例
def show_index_ticks(arr,index,pic_color,pic_label):
    plt.xlabel("Time (hour)")
    plt.ylabel("Traffic volume")
    #图例
    plt.plot(arr[index],color=pic_color,label=pic_label)
    plt.legend(loc='upper right')
    #改变x轴的刻度
    x_kedu=[0,4,8,12,16,20,24]
    orig_ticks = [i*20 for i in x_kedu]
    new_ticks = x_kedu
    plt.xticks(orig_ticks,new_ticks)
    #改变y轴的刻度
    y_kedu=[0,5,10,15,20]
    y_orig_ticks = y_kedu
    y_new_ticks =y_kedu
    plt.yticks(y_orig_ticks,y_new_ticks)
    plt.grid()
    plt.show()


def show_index_ticks_residual(arr,index,pic_color,pic_label):
    plt.xlabel("Time (hour)")
    plt.ylabel("Traffic volume")
    #图例
    plt.plot(arr[index],color=pic_color,label=pic_label)
    plt.legend(loc='upper right')
    #改变x轴的刻度
    x_kedu=[0,4,8,12,16,20,24]
    orig_ticks = [i*20 for i in x_kedu]
    new_ticks = x_kedu
    plt.xticks(orig_ticks,new_ticks)
    #residual 的 y轴的刻度
    y_kedu=[-2,-1,0,1,2]
    y_orig_ticks = y_kedu
    y_new_ticks =y_kedu
    plt.yticks(y_orig_ticks,y_new_ticks)
    #加上网格
    plt.grid()
    plt.show()

if __name__=="__main__":
    arr = myload("dump_arr_9-13.txt")
    main_x = myload("dump_main_x_9-13.txt")
    rest_x = myload("dump_rest_x_9-13.txt")
    #show1(arr)
    #show1(main_x)
    #show1(rest_x)
    #show2(main_x,rest_x)
    #show_index(arr,0)
    #show_index(main_x,0)
    #show_index(rest_x,0)

    show_index_ticks(arr,3,"black","traffic data flow")
    show_index_ticks(main_x,3,"red","main trend data")
    show_index_ticks_residual(rest_x,3,"blue","residual data")