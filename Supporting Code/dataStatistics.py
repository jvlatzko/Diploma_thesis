#!/usr/local/bin/python3.5

import numpy as np
import matplotlib.pyplot as plt

# dataPath = "ToF-data/Train_Val_Set/"
# imP = dataPath + "images/"; laP = dataPath + "labels/";

def getScenes(fname):
    if (fname.find("rain") > 0):
        scenes = [line.strip() for line in open(dataPath + "train.txt", 'r')]
    elif (fname.find("est") > 0): 
        scenes = [line.strip() for line in open(dataPath + "test.txt", 'r')]
    else: 
        print("Either training set or test set should be chosen.")
        raise ValueError("Noncompliant input: must be \"Train\" or \"Test\"") 
    print("%i scenes"%len(scenes))
    return scenes; 

def getAllScenes(dataPath = "./"):
    scenes = [line.strip() for line in open(dataPath + "allScenes.txt", 'r')]
    return scenes; 

def loglorentzian(x, s=1.0):
    return np.exp(-np.log(1+s*x**2))

def lorentzian(x,s=1.0):
    return np.log(1 + s * x**2)

def getVal(default):
    r = input().strip()
    if r == '':
        return default
    else:
        return r

print("Please enter path to data: [../wholeset/]")
dataPath = getVal("../wholeset/")
trainScenes = getScenes("Train"); 
allScenes = getAllScenes()

S, H, E = [], 0, 0
for i in range(len(allScenes)): 
    D = np.loadtxt("images/%s.dlm"%allScenes[i])
    G = np.loadtxt("labels/%s.dlm"%allScenes[i])
    diff = D[:, 8:1800:9] - G; 
    F = diff.flatten()
    S.append([np.min(diff), np.max(F), np.mean(F), np.median(F)])
    F -= F.mean()
    h, E = np.histogram(F, bins=1000, range=(-7.3, 7.3))
    H += h

T = np.array(S)
# plot error means # shows > 0, as expected for TOF
plt.hist(T[:,2], bins=30); plt.title('Histogram of error means'); plt.xlabel("Mean deviation [m]");  plt.show() 
ErrV = E[:-1]
# plot distribution of errors (as a pdf)
P = H/(np.sum(H)); 
P /= np.max(P) # needed to normalize likelihood distribution (otherwise, maximum is 0.09 = 9 % )
plt.plot(P); plt.show()
plt.semilogy(ErrV, P); plt.xlabel("Deviation [m]"); plt.ylabel("Normalised likelihood"); 
plt.savefig('Data-Error-Distribution_allScenes.png')
plt.savefig('Data-Error-Distribution_allScenes.pdf')

def displayErrHistograms(): 
    plt.figure(figsize=(12,4))
    for i,pl in enumerate((plt.semilogy, plt.plot)):
        plt.subplot(1,2,i+1)
        pl(ErrV, P)
        pl(ErrV, loglorentzian(ErrV, 50))
        plt.xlabel("Deviation [m]"); 
        plt.ylabel("Normalized likelihood")
    plt.show()
        
def displayErrHistrogramsZoomed():
    plt.figure(figsize=(12,4))
    b = 130; idx = np.r_[500-b:500+b]
    for i,pl in enumerate((plt.semilogy, plt.plot)):
        plt.subplot(1,2,i+1)
        pl(ErrV[idx],P[idx])
        pl(ErrV[idx], loglorentzian(ErrV[idx], 50))
        plt.xlabel("Deviation [m]"); 
        plt.ylabel("Normalized likelihood")
    plt.show()
    
def displayErrHistograms2(): 
    plt.figure(figsize=(12,4))
    for i,pl in enumerate((plt.semilogy, plt.plot)):
        plt.subplot(1,2,i+1)
        pl(ErrV, P, label="Errors")
    #    pl(ErrV, loglorentzian(ErrV,0.5), label="0.5")
        pl(ErrV, loglorentzian(ErrV,50), label="50")
        pl(ErrV, loglorentzian(ErrV,150), label="150")
        pl(ErrV, loglorentzian(ErrV,200), label="200")
    #    pl(ErrV, loglorentzian(ErrV,225), label="225")
    #    pl(ErrV, loglorentzian(ErrV,250), label="250")
        plt.legend()
    plt.show()
    
def displayErrHistograms2Zoomed(): 
    plt.figure(figsize=(12,4))
    b = 130; idx = np.r_[500-b:500+b]
    for i,pl in enumerate((plt.semilogy, plt.plot)):
        plt.subplot(1,2,i+1)
        pl(ErrV[idx], P[idx], label="Errors")
    #    pl(ErrV, loglorentzian(ErrV,0.5), label="0.5")
        pl(ErrV[idx], loglorentzian(ErrV[idx],50), label="50")
        pl(ErrV[idx], loglorentzian(ErrV[idx],150), label="150")
        pl(ErrV[idx], loglorentzian(ErrV[idx],200), label="200")
    #    pl(ErrV, loglorentzian(ErrV,225), label="225")
    #    pl(ErrV, loglorentzian(ErrV,250), label="250")
        plt.legend()
    plt.show()

displayErrHistograms()
plt.show()

