import numpy as np 


modelPath = "vgg19.npy"

wDict = np.load(modelPath, encoding = "bytes").item()

for name in wDict:
    for p in wDict[name]:
        print (p)