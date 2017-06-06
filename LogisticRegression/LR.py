import matplotlib.pyplot as plt
from numpy import *


def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet-LR.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)             #convert to NumPy matrix
    labelMat = mat(classLabels).transpose() #convert to NumPy matrix
    
    m,n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1))
    
    for k in range(maxCycles):              #heavy on matrix operations
        h = sigmoid(dataMatrix*weights)     #matrix mult
        error = (labelMat - h)              #vector subtraction
        weights = weights + alpha * dataMatrix.transpose()* error #matrix mult
    return weights

def GetResult():
    dataMat,labelMat=loadDataSet()
    weights=gradAscent(dataMat,labelMat)
    print (weights)
    plotBestFit(weights)
   
    
def plotBestFit(weights):
     
    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0] 
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)   
#    y=(0.48*x+4.12414)/(0.616)
#     y = (-weights[0]-weights[1]*x)/weights[2]  
    y = (-(float)(weights[0][0])-(float)(weights[1][0])*x)/(float)(weights[2][0])   

    ax.plot(x,y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()    
     
if __name__=='__main__':
    print ("ok")
    GetResult()
    print ("ok  2")
    
