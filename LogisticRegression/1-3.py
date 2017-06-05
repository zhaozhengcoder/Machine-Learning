import matplotlib.pyplot as plt
import numpy as np 
import math

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet-LR.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def plot_point(dataMat,labelMat):
    length=len(dataMat)
    xcord1=[]
    ycord1=[]
    xcord2=[]
    ycord2=[]
    for i in range(length):
        if labelMat[i]==1:
            xcord1.append(dataMat[i][0])
            ycord1.append(dataMat[i][1])
        else:
            xcord2.append(dataMat[i][0])
            ycord2.append(dataMat[i][1])
    plt.scatter(xcord1,ycord1,c='r')
    plt.scatter(xcord2,ycord2,c='g')

d,l=loadDataSet()
plot_point(d,l)
plt.show()