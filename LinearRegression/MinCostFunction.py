import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
from mpl_toolkits.mplot3d import Axes3D

fig=plt.figure()
ax=Axes3D(fig)


x=[1,2,3,4,5,6,7]
y=[4,7,10,13,16,19,22]


parameter0=np.arange(-10,10,0.2)
parameter1=np.arange(-10,10,0.2)

def func_j(p0,p1):
    sum=0
    for i in range(0,7):
        h=p0+p1*x[i]
        sum+=(h-y[i])**2
    sum=sum/14
    return sum

parameter0,parameter1=np.meshgrid(parameter0,parameter1)
z=func_j(parameter0,parameter1)
surf=ax.plot_surface(parameter0,parameter1,z)




min_value=np.min(z)
min_index=np.argmin(z)


print (np.unravel_index(min_index,z.shape))
min_point=np.unravel_index(min_index,z.shape)

min_x=min_point[0]
min_y=min_point[1]

print (parameter0[min_x][min_y])
print (parameter1[min_x][min_y])




plt.show()


