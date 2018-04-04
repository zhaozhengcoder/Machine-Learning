import numpy as np 
import pickle
import pca
#import tensorflow as tf

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


def split_dataset(dataset):
    trainX=[]
    trainY=[]
    trainX_len = 2  #使用3天预测一天
    trainY_len = 1  #使用3天预测一天
    days = dataset[0].shape[0]
    for road in dataset:
        trainX_per_road=[]
        trainY_pre_road=[]
        for i in range(0,days-(trainX_len+trainY_len-1)):
            trainX_per_road.append(road[i:i+trainX_len])
            trainY_pre_road.append(road[i+trainX_len:i+trainX_len+trainY_len])
        trainX.append(trainX_per_road)
        trainY.append(trainY_pre_road)
    return np.array(trainX),np.array(trainY)

def use_pca(dataset):
    dataset_rest=[]
    dataset_main=[]
    for road in dataset:
        pca_obj = pca.PCA(road,2)
        dataset_rest.append(pca_obj.rest_x)
        dataset_main.append(pca_obj.main_x)
    return dataset_main,dataset_rest

if __name__=="__main__":
    data = myload()  #data是一个list，里面是df格式的数据
    #transfer
    dataset = createdataset(data)  #dataset 的格式是 （路段 * 每一天 * 一天内的数据）20 * 7 * 480

    dataset_main,dataset_rest = use_pca(dataset)  # dataset_main = 路段 * 每一个路段里面的主成分 ; dataset_rest = 路段 * 每一个路段里面的偏差

    trainX,trainY = split_dataset(dataset_rest)

    print (trainX.shape)
    print (trainY.shape)