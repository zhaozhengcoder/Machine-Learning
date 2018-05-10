import numpy as np 
import pickle
import matplotlib.pyplot as plt
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
    filename ='dump2.txt'
    f = open(abs_path+filename,'rb')
    data =pickle.load(f)
    f.close()
    #print (data)   # 路段数 * 每个路段的信息（df的数据结构）
    return data


def split_time_nighttime(dataset):
    begin_time1=0
    end_time1=6*20  #6小时，每小时20个点
    begin_time2=21*20
    end_time2=24*20
    a1 = dataset[:,:,begin_time2:end_time2]     #(20, 14, 60)   #网上21点到24点
    a2 = dataset[:,:,begin_time1:end_time1]     #(20, 14, 120)  #早上0点到6点
    arr = np.concatenate((a1,a2),axis=2)        #(20, 14, 180)
    #print (arr.shape)
    return arr

def split_time_daytime(dataset):
    begin_time=6*20
    end_time = 21*20 
    arr = dataset[:,:,begin_time:end_time]     #(20, 14, 300)
    #print (arr.shape)
    return arr

"""
if __name__=="__main__":
    data = myload()                # data是一个list，里面是df格式的数据
    #transfer
    dataset = createdataset(data)  # dataset 的格式是 （路段 * 每一天 * 一天内的数据）20 * 14 * 480
    dataset = np.array(dataset)

    dataset_daytime=split_time_daytime(dataset)
    dataset_nighttime=split_time_nighttime(dataset)

    for i in range(10):
        plt.plot(dataset_daytime[i][0])
    plt.show()
    plt.close()

    for i in range(10):
        plt.plot(dataset_nighttime[i][0])
    plt.show()
    plt.close()
"""
    
      