"""
实现一个三层的神经网络 
一个输入层，一个输出层，隐层有3个结点

数据集同样也是tesetSet的数据集，和逻辑回归的数据集是同一个，格式如下：
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


#激活函数的倒数
def sigmod_diff(inX):
	return sigmod(inX) * (1-sigmod(inX))



def get_z1(inputs,mlabel,weights_layer1,b1,weights_layer2,b2):	
	z1=np.dot(weights_layer1,inputs)+b1
	return z1

def get_a1(inputs,mlabel,weights_layer1,b1,weights_layer2,b2):
	z1=np.dot(weights_layer1,inputs)+b1
	a1=sigmod(z1)
	return a1


def forward(inputs,mlabel,weights_layer1,b1,weights_layer2,b2):
	#从输入层到隐层
	z1=np.dot(weights_layer1,inputs)+b1
	a1=sigmod(z1)

	#从隐层到输出层
	z2=np.dot(weights_layer2,a1)+b2
	a2=sigmod(z2)	
	
	#error
	dz2=a2-mlabel
	return dz2

#计算cost，每一次迭代之后，都算一下cost，看看cost是否在减小
def cost(inputs,mlabel,weights_layer1,b1,weights_layer2,b2):
	nx,m=inputs.shape
	#从输入层到隐层
	z1=np.dot(weights_layer1,inputs)+b1
	a1=sigmod(z1)

	#从隐层到输出层
	z2=np.dot(weights_layer2,a1)+b2
	a2=sigmod(z2)	
	
	#cost
	cost=-mlabel* np.log(a2)-(a2-mlabel)*np.log(1-a2)
	return np.sum(cost)/m


#将训练的输出和真实的结果show出来
def show1(inputs,mlabel,weights_layer1,b1,weights_layer2,b2):
	nx,m=inputs.shape
	#从输入层到隐层
	z1=np.dot(weights_layer1,inputs)+b1
	a1=sigmod(z1)

	#从隐层到输出层
	z2=np.dot(weights_layer2,a1)+b2
	a2=sigmod(z2)	
	
	plt.plot(mlabel)
	plt.plot(a2[0])
	plt.show()
	

def show2(inputs,mlabel,weights_layer1,b1,weights_layer2,b2):
	nx,m=inputs.shape
	#从输入层到隐层
	z1=np.dot(weights_layer1,inputs)+b1
	a1=sigmod(z1)

	#从隐层到输出层
	z2=np.dot(weights_layer2,a1)+b2
	a2=sigmod(z2)	
	
	new_a2=[]
	for i in a2[0]:
		#这里用0.1和0.9，是为了避免和mlabel画出来的线重合
		if i <0.5:
			new_a2.append(0.1)
		if i>=0.5:
			new_a2.append(0.9)
	
	plt.plot(mlabel)
	plt.plot(new_a2)
	plt.show()



#正向传播和反向传播
def gradientdesc(mdata,mlabel,weights_layer1,b1,weights_layer2,b2):
	nx,m=mdata.shape
	#调用正向传播的函数，得到dz2
	dz2=forward(mdata,mlabel,weights_layer1,b1,weights_layer2,b2)
	
	#求dw2和db2
	a1=get_a1(mdata,mlabel,weights_layer1,b1,weights_layer2,b2)
	dw2 = (1/float(m)) * np.dot(dz2,a1.T)
	db2 = (1/float(m)) * np.sum(dz2)

	#求dw1和db1
	z1=get_z1(mdata,mlabel,weights_layer1,b1,weights_layer2,b2)
	dz1 =np.dot(weights_layer2.T,dz2) * sigmod_diff(z1)
	
	dw1 = (1/float(m)) * np.dot(dz1,mdata.T)
	db1 = (1/float(m)) * np.sum(dz1)	
	
	#更新w1,w2,b1,b2
	weights_layer1=weights_layer1 - alpha * dw1
	weights_layer2=weights_layer2 - alpha * dw2
	b1=b1-alpha*db1
	b2=b2-alpha*db2

	return weights_layer1,b1,weights_layer2,b2


def three_layer_nn(maxcycle=5000):
	mdata,mlabel=loadDataset()
	nx,m=mdata.shape	

	hiden_node=3

	#随机初始化 权值矩阵
	weights_layer1=random.random(size=(hiden_node,nx))
	b1=random.random(size=(hiden_node,m))

	weights_layer2=random.random(size=(1,hiden_node))
	b2=random.random(size=(1,m))

	#迭代	
	for i in range(maxcycle):
		weights_layer1,b1,weights_layer2,b2=gradientdesc(mdata,mlabel,weights_layer1,b1,weights_layer2,b2)
		print (cost(mdata,mlabel,weights_layer1,b1,weights_layer2,b2))


	#show
	show2(mdata,mlabel,weights_layer1,b1,weights_layer2,b2)	

if __name__=='__main__':
	maxcycle=15000
	three_layer_nn(maxcycle)

