"""
实现一个逻辑回归

1. 数据集是testSet，机器学习实战里面的一个数据集，格式如下:
   x1           x2		y
   -0.017612	14.053064	0
   ....


"""

import numpy as np
from numpy import random
import matplotlib.pyplot as plt

alpha=0.01


#加载数据集，原来的数据在文件排列是按行排列
#为了计算需要，将原来的数据加载到了矩阵之后，给矩阵装置了，是数据变成按列排列
def loadDataset():
	data=[]
	label=[]
	f=open("textSet.txt")
	for line in f:
		lineArr=line.strip().split()
		data.append( [float(lineArr[0]),float(lineArr[1]) ] ) 
		label.append(float(lineArr[2]))
	mdata=np.array(data)
	mlabel=np.array(label)
	return mdata.T,mlabel.T


def sigmod(inX):
	return 1.0/(1+np.exp(-inX))

#计算error，也就是dz，这个error 是为了计算梯度下降
def forward(mdata,mlabel,weight,b):
	z=np.dot(weight,mdata)+b		
	a=sigmod(z)
	error= a - mlabel
	return error

#梯度下降，计算出dw，db，然后更新w和b
def gradDesc(mdata,error,weight,b):
	nx,m=mdata.shape
	dw=(1/m)*np.dot(mdata,error.T)
	db=(1/m)*np.sum(error)
	weight_transpose = weight.T - alpha*dw
	b=b-alpha*db
	return weight_transpose.T,b

#代价函数，写这个函数的目的是，在迭代的时候，输出每一次迭代后的cost，判断cost是否是在下降
def cost(mdata,mlabel,weight,b):	
	nx,m=mdata.shape
	z=np.dot(weight,mdata)+b		
	a=sigmod(z)
	cost=-mlabel*np.log(a)-(a-mlabel)*np.log(1-a)
	return np.sum(cost)/m

#show result
def show1(mdata,mlabel,weight,b):	
	nx,m=mdata.shape
	z=np.dot(weight,mdata)+b		
	a=sigmod(z)
	for i,j in zip(a[0],mlabel):
		print (i,' , ',j)

#将原始的数据和计算之后得到的数据对比，以折线图的方式显示
def show2(mdata,mlabel,weight,b):	
	nx,m=mdata.shape
	z=np.dot(weight,mdata)+b		
	a=sigmod(z)
	plt.plot(a[0])
	plt.plot(mlabel)
	plt.show()


#将计算得到的数据二值化，小于0.5的变成0，大于0.5的变成1
#由于绝大多数的点都是相同的，所以很多点会被覆盖
def show3(mdata,mlabel,weight,b):	
	nx,m=mdata.shape
	z=np.dot(weight,mdata)+b		
	a=sigmod(z)

	a2=[]	
	for i in a[0]:
		if i >0.5:
			a2.append(1)
		if i<=0.5:
			a2.append(0)
	plt.plot(a2,'.')
	plt.plot(mlabel,'.')
	plt.show()

def regress(maxcycle=100):
	mdata,mlabel=loadDataset()
	nx,m=mdata.shape

	#w和b 随机初始化，代码的目的就是求w和b
	weight=random.random(size=(1,nx))
	b=random.random(size=(1,m))
	
	#迭代
	for i in range(maxcycle):
		error=forward(mdata,mlabel,weight,b)
		weight,b=gradDesc(mdata,error,weight,b)
		print (cost(mdata,mlabel,weight,b))
	
	show1(mdata,mlabel,weight,b)
	show2(mdata,mlabel,weight,b)
	show3(mdata,mlabel,weight,b)

if __name__=='__main__':
	regress(3000)
