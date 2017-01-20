#!/usr/local/bin/python3.5

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from skimage import img_as_float, io

#c0 = 299758492
c0 = 3*10**8
TOFfreq = 20*10**6

import os
cwd = os.getcwd()
print(cwd)

path = "depth_data/ToF-data/Train_65/raw/"
outImage = "depth_data/ToF-data/Train_65/images/"
outLabel = "depth_data/ToF-data/Train_65/labels/"
name = "bathroom_03"

# load raw frame
r0 = np.loadtxt(path + name + '_raw_0.txt', delimiter=',')
r1 = np.loadtxt(path + name + '_raw_1.txt', delimiter=',')
r2 = np.loadtxt(path + name + '_raw_2.txt', delimiter=',')
r3 = np.loadtxt(path + name + '_raw_3.txt', delimiter=',')
# Will serve as Channels 1 through 4
C0 = img_as_float(r0)
C1 = img_as_float(r1)
C2 = img_as_float(r2)
C3 = img_as_float(r3)

# load ground truth
GT = np.loadtxt(path + name + '_gt.txt', delimiter=',')

# Will serve as Channels 5 and 6
Nom = C3 - C1
Denom = C0 - C2
# Will serve as Channel 7
Quot = Nom/Denom; 
# Will serve as Channel 9
P = np.arctan2(Nom, Denom)

D = P.copy()
D[D < 0] += (2*np.pi)
D *= c0/(4*np.pi*TOFfreq)

# Intensity, will serve as channel 8
I = 1/4.0*(C0+C1+C2+C3)

# Need to project to/from plane
# From camera matrix file: 
cx = 100.5; cy = 100.5; fx = 317.16; fy = 317.16; 
shp = P.shape; xc, yc = np.meshgrid(range(shp[0]), range(shp[1])); 
# convert radial to z distance
xx = (xc - cx) / fx; yy = (yc - cy) / fy; 

T = np.dstack((C0, C1, C2, C3, Nom, Denom, Quot, I, D)); 

Cmbd = np.ones((T.shape[0], T.shape[2]*T.shape[1])); 

for i in range(0, 200): 
    for j in range(0, 200): 
        Cmbd[i, (9*j):(9*j+9)] = T[i, j];

# save as dlm file, using tabs as delimiter
np.savetxt(outImage + name + ".dlm", Cmbd, fmt='%.5e', delimiter='\t')

# Project GT to sphere (z depth to radial distance)
xxd = xx*GT; yyd = yy*GT; 
GTz = np.sqrt(np.square(xxd) + np.square(yyd) + np.square(GT)); 

np.savetxt(outLabel + name + ".dlm", GTz, fmt='%.5e', delimiter='\t')
print("Saved all columns for %s in file!"%name)
