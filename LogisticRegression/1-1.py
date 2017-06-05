import matplotlib.pyplot as plt
import numpy as np 
import math

alpha=0.01


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
    

def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))



def fun_z(th0,th1,th2,x1,x2):
    return th0+ th1*x1+th2*x2

def fun_h(z):
    return 1.0/(1+math.exp(-z))

def plot_line(theta0,theta1,theta2):
    x=np.arange(-5,5,0.1)
    #y=theta0+theta1*x
    y= (theta0+theta1*x)/(-theta2)
    plt.plot(x,y)
    plt.show()


def gradAscent(dataMat,labelMat):
    theta0=np.random.normal()
    theta1=np.random.normal()
    theta2=np.random.normal()
    
    m=len(dataMat)
    for times in range(3000):
        sum1=0.0
        sum2=0.0
        sum3=0.0
        for i in range(m):
            z=fun_z(theta0,theta1,theta2,dataMat[i][0],dataMat[i][1])

            sum1=sum1+(fun_h(z)-labelMat[i])
            sum2=sum2+(fun_h(z)-labelMat[i])*dataMat[i][0]
            sum3=sum3+(fun_h(z)-labelMat[i])*dataMat[i][1]
        theta0=theta0-(alpha*sum1)
        theta1=theta1-(alpha*sum2)
        theta2=theta2-(alpha*sum3)
    return theta0,theta1,theta2

d,l=loadDataSet()
th0,th1,th2=gradAscent(d,l)

print (th0,"  ,  ",th1,"  ,  ",th2)

plot_point(d,l)
plot_line(th0,th1,th2)

