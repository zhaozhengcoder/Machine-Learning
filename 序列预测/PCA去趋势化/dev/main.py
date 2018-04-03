import numpy as np 
import pickle
import tensorflow as tf

abs_path='C:/Users/wwwa8/Documents/GitHub/Machine-Learning/序列预测/PCA去趋势化/dev/'


#原始的data里面的数据格式是dataframe，arr改成了里面也是list
def transfer(data):
    vol_col_index = 1 # 找到流量对应的列
    height = len(data)
    width = data[0].shape[0]

    arr = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            arr[i,j]=data[i].iloc[j,vol_col_index]
    return arr


def createdataset(data):
    dataset=[]
    for road in data:     #对于某一条路的数据
        dataset.append(transfer(road))
    return dataset 

def myload():
    filename ='dump.txt'
    f = open(abs_path+filename,'rb')
    data =pickle.load(f)
    f.close()
    #print (data)   # 路段数 * 每个路段的信息（df的数据结构）
    return data


def split_dataset(arr):
    trainX=[]
    trainY=[]
    trainX_len = 2 #使用3天预测一天
    trainY_len = 1 #使用3天预测一天
    day = 24*60
    merge_step = 3 
    daylen =day/merge_step

    days = arr.shape[1]/daylen  #总天数
    for i in range(0,days-(trainX_len+trainY_len-1)):
        trainX.append(arr[ i*daylen            :(i+trainX_len)*daylen]           )
        trainY.append(arr[(i+trainX_len)*daylen:(i+trainX_len+trainY_len)*daylen])
    return trainX,trainY
    

if __name__=="__main__":
    data = myload()
    #transfer
    dataset = createdataset(data)  #dataset 的格式是 （路段 * 每一天 * 一天内的数据）