import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import mean_squared_error

"""
from keras.models import Sequential 
from keras.layers import Dense 
from keras.layers import LSTM 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import mean_squared_error
"""

#从文件加载数据
def load_data():
    path='123.csv'
    # load the dataset
    dataframe = read_csv(path, usecols=[0], engine='python', skipfooter=0)
    dataset = dataframe.values
    # 将整型变为float
    dataset = dataset.astype('float32')
    ## show
    #print (dataset)
    print ("shape : ",dataset.shape)
    #plt.plot(dataset,'.')
    #plt.show()
    return dataset


def split_data(dataset):
    # split into train and test sets
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    return train,test

"""
create_dataset 的作用是，划分数据 
原始的dataset = [100,200,300,400,500,600,700,800]
dataX = 
[[ 100.  200.  300.]
 [ 200.  300.  400.]
 [ 300.  400.  500.]
 [ 400.  500.  600.]
 [ 500.  600.  700.]
 [ 600.  700.  800.]]
dataY = 
[ 400.  500.  600.  700.  800.  900.]
"""
def create_dataset(dataset,look_back=3):
    dataX, dataY = [], [] 
    for i in range(len(dataset)-look_back-1): 
        a = dataset[i:(i+look_back), 0] 
        dataX.append(a) 
        dataY.append(dataset[i + look_back, 0]) 
    return np.array(dataX), np.array(dataY)



if __name__=="__main__":
    dataset=load_data()
 
    #标准化
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    
    train_data,test_data=split_data(dataset)
    train_x,train_y=create_dataset(train_data)
    test_x,test_y =create_dataset(test_data)

    #reshape 
    train_y=np.reshape(train_y,(len(train_y),-1))
    test_y=np.reshape(test_y,(len(test_y),-1))

    print (train_x)
    print (train_y)

    print ("train_x.shape : ",train_x.shape)
    print ("train_y.shape : ",train_y.shape)
    print ("test_x.shape : ",test_x.shape)
    print ("test_y.shape : ",test_y.shape)
    """
    train_x.shape :  (1261, 3)
    train_y.shape :  (1261, 1)  
    test_x.shape :  (620, 3)
    test_y.shape :  (620, 1)
    """

    #todo svr method 
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    y_rbf = svr_rbf.fit(train_x,train_y).predict(test_x)
    #y_lin = svr_lin.fit(X, y).predict(X)
    #y_poly = svr_poly.fit(X, y).predict(X)

    #反标准化
    y_rbf=np.reshape(y_rbf,(len(y_rbf),-1))
    y_rbf=scaler.inverse_transform(y_rbf)
    test_y=scaler.inverse_transform(test_y)
    
    plt.plot(y_rbf)
    plt.plot(test_y)
    plt.show()

    #print (dataset)

    #rmse
    testScore = math.sqrt(mean_squared_error(y_rbf,test_y))
    print('Test Score: %.2f RMSE' % (testScore))
    ## Test Score: 11.58 RMSE
