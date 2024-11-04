import numpy as np
import pandas as pd

phagenamelist = pd.read_csv('PHD-data-work3/phd-4723-V-name.csv', header=None)
hostnamelist = pd.read_csv('PHD-data-work3/phd-553-H-name.csv', header=None)
rawA = pd.read_csv('PHD-data-work3/PHD-pairs.csv', header=None)

print(phagenamelist.shape)
print(hostnamelist.shape)
print(rawA.shape)

num_p = phagenamelist.shape[0]
num_h = hostnamelist.shape[0]

newA = np.zeros((num_p,num_h))

num_i = rawA.shape[0]
for i in range(num_i):
    ip = np.array(phagenamelist)[:, 0].tolist().index(rawA.iloc[i, 0])
    ih = np.array(hostnamelist)[:, 0].tolist().index(rawA.iloc[i, 1])
    newA[ip, ih] = 1

print(newA)
print(newA.sum())
print(newA.sum(axis=0))
np.savetxt("PHD-data-work3/ICTV-VH.csv", newA, fmt = '%d', delimiter=",")