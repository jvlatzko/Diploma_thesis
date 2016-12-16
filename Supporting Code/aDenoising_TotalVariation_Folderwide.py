#!/usr/local/bin/python3.5

import numpy as np
import matplotlib.pyplot as plt
import time

from skimage import img_as_float, measure
from skimage.restoration import denoise_tv_chambolle

#from os.path import isfile, join
testSetLoss = 0.47842418078290611; 

def getScenes(fname):
    if (fname == "Train"):
        scenes = [line.strip() for line in open(dataPath + "train.txt", 'r')]
    elif (fname =="Test"): 
        scenes = [line.strip() for line in open(dataPath + "test.txt", 'r')]
    else: 
        print("Either training set or test set should be chosen.")
        raise ValueError("Noncompliant input: must be \"Train\" or \"Test\"") 
    return scenes; 

def Lorentzian(Img, s = 50):
    return np.log(1.0 + s * Img**2)

def LorentzianLoss(Img, Ground, s = 50): 
    tImg = Img - Ground; 
    tImg -= np.mean(tImg); 
    return np.mean(np.log(1.0 + s*(tImg)**2))

def shiftedMSE(Img, Ground, drop, bool):
    mean = 0
    
    shift = 0; 
    if( bool ):     
        shift = np.mean(Img - Ground);
    cx, cy = Img.shape;
    count = 0
    tot = 0.0
    for i in range(cy):
        rowMean = 0
        for j in range(cx):
            curr = (Img[i,j] - shift - Ground[i,j])
            if( np.abs(curr) < drop):
                rowMean += curr**2;
            else:
                rowMean += drop**2
                count += 1
                tot += np.abs(curr)
        mean += rowMean;
    mean /= (cx*cy)
    # print("Dropped %i values in MSE calculation, summing to %f squared"%(count, tot))
    # print("shiftedMSE: ", mean)
    return mean;

def getVal(default): 
    r = input().strip();
    if( r == ''):
        return default
    else:
        return r

def evaluateScenes(sceneSet, weight):
    totalLoss = 0.0;
    i = 0; 
    for scene in sceneSet:
        gt, formula, C0, C1, C2, C3 = readResults(scene); 
        GT[i, :, :] = gt; CD[i, :, :] = formula; 
        formLoss = LorentzianLoss(gt, formula);
        totalFormLoss.append(formLoss); 
        Dd = denoise_tv_chambolle(formula, weight, multichannel=False)
        CDd[i, :, :] = Dd.copy();     
        totalLoss += LorentzianLoss(Dd, gt); 
        i += 1
    #    print(LorentzianLoss(Dd, gt)); 
    totalLoss /= len(sceneSet)
    print("Result for weight %f : %f vs. %f"%(weight, totalLoss, testSetLoss)); 
    return totalLoss
    
def readResults(scene_name):
    groundTruth = np.loadtxt(dataPath + "labels/" + scene_name + ".dlm", delimiter='\t') # Get GT from data folder
    allInputRTF = np.loadtxt(dataPath + "images/" + scene_name + ".dlm", delimiter='\t') # Get INPT from data folder
    formula = allInputRTF[:, 8:1800:9]
    C0 = allInputRTF[:, 0:1800:9]
    C1 = allInputRTF[:, 1:1800:9]
    C2 = allInputRTF[:, 2:1800:9]
    C3 = allInputRTF[:, 3:1800:9]
    return groundTruth, formula, C0, C1, C2, C3


def totVarApplication(weight, sceneSet):
    grid = np.zeros((len(weight))); 
    for i in range(len(weight)):
        grid[i] = evaluateScenes(sceneSet, weight[i]); 
    return grid; 

print("Please enter path to data: ")
dataPath = getVal("../depth_data/ToF-data/Train_Val_Set/")
#dataPath = "../depth_data/ToF-data/Train_Val_Set/"

testScenes  = getScenes("Test")
print("Got test")

GT      = np.ndarray((len(testScenes), 200, 200))
CD      = np.ndarray((len(testScenes), 200, 200))
CDd      = np.ndarray((len(testScenes), 200, 200))

gt, f, c0t, c1t, c2t, c3t = readResults(testScenes[0]); 
plt.viridis(); 
plt.imshow(gt); plt.colorbar(); plt.show()

print("Entering denoising part\n")
# DENOISING PART

minWeight        = 0.01; 
maxWeight        = 0.9; 
step             = 0.01
weight           = np.arange(minWeight, maxWeight, step);
totalFormLoss   = []; 

t = time.time()
lossTV = totVarApplication(weight, testScenes); 
print("DONE")
print("(after %f s)"%(time.time() - t)); 


#GT = np.empty((int(numberOfScenes), 1)); 


plt.plot(weight, lossLorentzianShift[:, 1]); plt.show()
plt.plot(weight, lossMSEshift[:, 1]); plt.show()
#plt.plot(weight, lossLorentzianShift[:, 0]); plt.show()
#plt.plot(weight, lossLorentzianShift[:, 2]); plt.show()
# np.savetxt("MSE.txt"+path, outData, delimiter='\t', fmt='%.5e')
