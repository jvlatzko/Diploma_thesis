#!/usr/local/bin/python3.5

import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float, io

from os import listdir, path, getcwd
cwd = getcwd()
print(cwd)

#c0 = 299758492 #Simulation by luxrender is not as exact - do not introduce new "error"
c0 = 3*10**8
TOFfreq = 20*10**6

dataPath = "ToF-data/Validate_10/raw/"
#dataPath = "data/nice_data/"
outPathImages   = "ToF-data/Validate_10/images/"
outPathLabels   = "ToF-data/Validate_10/labels/"
files = sorted(listdir(dataPath))

numberOfScenes = (len(files))/5; # macOS has .DS_store

for i in np.arange(0, int(numberOfScenes), 1): 
    sceneNameT = files[0+i*5].split(".")[0]
    if(not sceneNameT.count("gt")): 
        print("Problem with files (hidden files?). Only raw frames in folder expected"); 
    sceneName = sceneNameT[0:(len(sceneNameT)-3)] #remove "_gt"
    print(sceneName)

    # load noisy raw frames
    r0 = np.loadtxt(dataPath + sceneName + '_raw_0.txt', delimiter=',')
    r1 = np.loadtxt(dataPath + sceneName + '_raw_1.txt', delimiter=',')
    r2 = np.loadtxt(dataPath + sceneName + '_raw_2.txt', delimiter=',')
    r3 = np.loadtxt(dataPath + sceneName + '_raw_3.txt', delimiter=',')
    # Will serve as Channels 1 through 4
    C0 = img_as_float(r0)
    C1 = img_as_float(r1)
    C2 = img_as_float(r2)
    C3 = img_as_float(r3)

    # load ground truth - this is true z depth as opposed to other data
    # needs to be saved as label
    # might be added as channel for troubleshooting
    GT = np.loadtxt(dataPath + sceneName + '_gt.txt', delimiter=',')

    # Will serve as Channels 5 and 6
    Nom = C3 - C1
    Denom = C0 - C2
    # Will serve as Channel 7
    Quot = np.ma.divide(Nom, Denom); 
    mm = np.max(Quot); 
    Quot[Quot.mask] = mm; 
    Q = Quot.data;
    # Intensity, will serve as channel 8
    I = 1/4.0*(C0+C1+C2+C3)
    # Will serve as Channel 9
    P = np.arctan2(Nom, Denom)
    # Variant: shift phase as one would shift depth
    D = P.copy()
    D[D < 0] += (2*np.pi)
    D *= c0/(4*np.pi*TOFfreq)

    # D = P.copy()
    # D *= c0/(4*np.pi*TOFfreq)
    
    # project to plane
    # From camera matrix file: 
    cx = 100.5; cy = 100.5; fx = 317.16; fy = 317.16; 
    shp = P.shape; xc, yc = np.meshgrid(range(shp[0]), range(shp[1])); 
    xx = (xc - cx) / fx; yy = (yc - cy) / fy; 
    # convert radial to z distance
    # D /= np.sqrt(np.square(xx) + np.square(yy) + 1.0)

    # Concatenate all 
    T = np.dstack((C0, C1, C2, C3, Nom, Denom, Q, I, D)); 

    Cmbd = np.ones((T.shape[0], T.shape[2]*T.shape[1])); 

    for i in range(0, 200): 
        for j in range(0, 200): 
            Cmbd[i, (9*j):(9*j+9)] = T[i, j];


    # save as dlm file, using tabs as delimiter
    np.savetxt(outPathImages + sceneName + '.dlm', Cmbd, fmt='%.5e', delimiter='\t')

    # Project GT to sphere (z depth to radial distance)
    xxd = xx*GT; yyd = yy*GT; 
    GTz = np.sqrt(np.square(xxd) + np.square(yyd) + np.square(GT)); 
    np.savetxt(outPathLabels + sceneName + '.dlm', GTz, fmt='%.5e', delimiter='\t')
    print("Saved all columns for input and label for scene %s!"%sceneName)
