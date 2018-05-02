import matplotlib.pyplot as plt
import numpy as np

a1 = np.random.randn(5)
a2 = np.random.randn(5)

plt.plot(a1,color='red',label='red')
plt.plot(a2,color='blue',label='blue')
plt.legend(loc='upper right')
plt.show()