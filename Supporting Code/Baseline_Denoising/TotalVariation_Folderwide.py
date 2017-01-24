#!/usr/local/bin/python3.5


import numpy as np
import random, re
from skimage import img_as_float, measure
from skimage.restoration import denoise_tv_chambolle
import matplotlib.pyplot as plt


def getScenes(fname, dataPath = "./"):
    if (fname == "Train"):
        scenes = [line.strip() for line in open(dataPath + "train.txt", 'r')]
    elif (fname =="Validation"): 
        scenes = [line.strip() for line in open(dataPath + "test.txt", 'r')]
    elif (fname == "Test"):
        scenes = [line.strip() for line in open(dataPath + "trueTest.txt", 'r')]
    else: 
        print("Either validation set or test set should be chosen.")
        raise ValueError("Noncompliant input: must be \"Train\" or \"Validate\" or \"Test\"") 
    return scenes; 

def getAllScenes(dataPath = "./"):
    scenes = [line.strip() for line in open(dataPath + "allScenes.txt", 'r')]
    return scenes; 

def lorentzian(Img, s = 50):
    return np.log(1.0 + s * Img**2)

def lorentzianLoss(Img, Ground, s = 50): 
    tImg = Img - Ground; 
    tImg -= np.mean(tImg); 
    return np.mean(np.log(1.0 + s*(tImg)**2))

def myMSE(Img, Ground):
    tImg = Img - Ground; 
    tImg -= np.mean(tImg); 
    return np.mean( np.sqrt(np.square(tImg)) )

def loadScene(scene_name, dataPath = "./"):
    groundTruth = np.loadtxt(dataPath + "labels/" + scene_name + ".dlm", delimiter='\t') # Get GT from data folder
    # GT DONE
    allInputRTF = np.loadtxt(dataPath + "images/" + scene_name + ".dlm", delimiter='\t') # Get INPT from data folder
    depthD = allInputRTF[:, 8:1800:9]
    # INPUT DONE
    return groundTruth, depthD

def storeLoss(sceneset):
    ddict = {}
    for scene in sceneset:
        td, tgt = loadScene(scene); 
        tloss = lorentzianLoss(td, tgt); 
        ddict[scene] = tloss; 
    return ddict
    
def evaluateScenes(sceneSet, weight, ddict):
    totalLoss = 0.0; totalErr = 0.0; 
    testSetLoss = np.mean(list(ddict.values()))
    #print("Filter config: Weight %f"%(weight))
    for scene in sceneSet:
        gt, formula = loadScene(scene); 
        formLoss = ddict[scene];
        Dd = denoise_tv_chambolle(formula, weight, multichannel=False)
        totalLoss += lorentzianLoss(Dd, gt); 
        totalErr += myMSE(Dd, gt)
    #    print(LorentzianLoss(Dd, gt)); 
    totalLoss /= len(sceneSet)
    totalErr /= len(sceneSet)
    print("Result for weight %f : %f vs. %f"%(weight, totalLoss, testSetLoss)); 
    return totalLoss, totalErr

def displayScene(scene, weight, ddict):
    gt, formula = loadScene(scene); 
    formLoss = ddict[scene];
    Dd = denoise_tv_chambolle(formula, weight, multichannel=False)
    plt.imshow(Dd); plt.axis('off'); plt.colorbar(); plt.show(); 

def visualizeScene(scene, weight, deltaSwitch = True):
    gt, formula = loadScene(scene); 
    Dd = denoise_tv_chambolle(formula, weight, multichannel=False)
    vmin, vmax = np.min(gt), np.max(gt); 
    plt.figure(figsize=(14,3)); 
    plt.subplot(131); plt.imshow(formula, interpolation='nearest', vmin=vmin, vmax=vmax); plt.title("Baseline (%s)"%float('%.5g' % lorentzianLoss(formula, gt)) ); plt.colorbar(); plt.axis('off'); 
    plt.subplot(132); plt.imshow(Dd, interpolation='nearest', vmin=vmin, vmax=vmax); plt.title("TV (%s)"%float('%.5g' % lorentzianLoss(Dd, gt)) ); plt.colorbar(); plt.axis('off'); 
    plt.subplot(133); plt.imshow(gt, interpolation='nearest', vmin=vmin, vmax=vmax); plt.title("Ground Truth %s"%scene); plt.colorbar(); plt.axis('off'); 
    if( deltaSwitch): 
        plt.figure(figsize=(14,3)); 
        deltaIn = formula - gt; deltaIn -= np.mean(deltaIn); deltaIn = lorentzian(deltaIn); 
        deltaOut = Dd - gt; deltaOut -= np.mean(deltaOut); deltaOut = lorentzian(deltaOut); 
        improv = deltaOut - deltaIn; vmindel = np.min(improv); vmaxdel = np.max(improv); 
        vmin, vmax = np.min((np.min(deltaIn), np.min(deltaOut))), np.max((np.max(deltaIn), np.max(deltaOut))); 
        plt.subplot(131); plt.imshow(deltaIn, interpolation='nearest', vmin=vmin, vmax=vmax); plt.title("Loss Input"); plt.colorbar(); plt.axis('off'); 
        plt.subplot(132); plt.imshow(deltaOut, interpolation='nearest', vmin=vmin, vmax=vmax); plt.title("Loss Output"); plt.colorbar();plt.axis('off'); 
        plt.subplot(133); plt.imshow(improv, interpolation='nearest', vmin = vmindel, vmax = vmaxdel); plt.title("Difference"); plt.colorbar(); plt.axis('off'); 
    plt.show()

def saveScene(scene, weight, deltaSwitch = True):
    gt, formula = loadScene(scene); 
    Dd = denoise_tv_chambolle(formula, weight, multichannel=False)
    vmin, vmax = np.min(gt), np.max(gt); 
    plt.figure(figsize=(14,3)); 
    plt.subplot(131); plt.imshow(formula, interpolation='nearest', vmin=vmin, vmax=vmax); plt.title("Baseline (%s)"%float('%.5g' % lorentzianLoss(formula, gt)) ); plt.colorbar(); plt.axis('off'); 
    plt.subplot(132); plt.imshow(Dd, interpolation='nearest', vmin=vmin, vmax=vmax); plt.title("TV (%s)"%float('%.5g' % lorentzianLoss(Dd, gt)) ); plt.colorbar(); plt.axis('off'); 
    plt.subplot(133); plt.imshow(gt, interpolation='nearest', vmin=vmin, vmax=vmax); plt.title("Ground Truth %s"%scene); plt.axis('off'); plt.colorbar(); 
    plt.savefig('%s_TV.png'%scene, bbox_inches='tight')
    if( deltaSwitch): 
        plt.figure(figsize=(14,3)); 
        deltaIn = formula - gt; deltaIn -= np.mean(deltaIn); deltaIn = lorentzian(deltaIn); 
        deltaOut = Dd - gt; deltaOut -= np.mean(deltaOut); deltaOut = lorentzian(deltaOut); 
        improv = deltaOut - deltaIn; vmindel = np.min(improv); vmaxdel = np.max(improv); 
        vmin, vmax = np.min((np.min(deltaIn), np.min(deltaOut))), np.max((np.max(deltaIn), np.max(deltaOut))); 
        plt.subplot(131); plt.imshow(deltaIn, interpolation='nearest', vmin=vmin, vmax=vmax); plt.title("Loss Input"); plt.colorbar(); plt.axis('off'); 
        plt.subplot(132); plt.imshow(deltaOut, interpolation='nearest', vmin=vmin, vmax=vmax); plt.title("Loss Output"); plt.colorbar(); plt.axis('off'); 
        plt.subplot(133); plt.imshow(improv, interpolation='nearest', vmin = vmindel, vmax = vmaxdel); plt.title("Difference"); plt.axis('off');  plt.colorbar(); 
        plt.savefig('%s_TV_delta.png'%scene, bbox_inches='tight')
    plt.close()

def totVarApplication(startWeight, stopWeight, stepWeight, sceneSet, sceneSetLoss):
    weight = np.arange(startWeight, stopWeight, stepWeight);
    grid = np.zeros((len(weight))); 
    for i in range(len(weight)):
        grid[i], set_MSE = evaluateScenes(sceneSet, weight[i], sceneSetLoss); 
    return weight, grid; 

def saveBadScene(scene, weight):
    gt, formula = loadScene(scene); 
    Dd = denoise_tv_chambolle(formula, weight, multichannel=False)
    vmin, vmax = np.min(gt), np.max(gt); 
    plt.figure(figsize=(14,3)); 
    plt.subplot(131); plt.imshow(formula, interpolation='nearest', vmin=vmin, vmax=vmax); plt.title("Baseline"); plt.colorbar(); plt.axis('off'); 
    plt.subplot(132); plt.imshow(Dd, interpolation='nearest', vmin=vmin, vmax=vmax); plt.title("Total Variation"); plt.colorbar(); plt.axis('off'); 
    plt.subplot(133); plt.imshow(gt, interpolation='nearest', vmin=vmin, vmax=vmax); plt.title("Ground Truth %s"%scene); plt.axis('off'); plt.colorbar(); 
    plt.savefig('%s_TV_bad.png'%scene, bbox_inches='tight')



plt.viridis(); 
allScenes = getAllScenes(); 
valScenes = getScenes("Validation")
testScenes = getScenes("Test")
#valLoss = storeLoss(valScenes)
#testLoss = storeLoss(testScenes)

#weights, grid = totVarApplication(0.1, 0.2, 0.05, valScenes, valLoss); 
saveScene("bathroom_29", 0.15)
#saveBadScene("bathroom_29", 2.4)


#plt.plot(weights, grid, label="Total Variation"); plt.xlabel("TV weight"); plt.ylabel("TV loss"); plt.show(); 

#losses = storeLoss(allScenes) 
# Previous result
testSetLoss = 0.41180973742128724; 

# In [25]: weights, grid = totVarApplication(0.1, 0.21, 0.01, valScenes, valLoss);
# Result for weight 0.100000 : 0.402317 vs. 0.411810
# Result for weight 0.110000 : 0.402136 vs. 0.411810
# Result for weight 0.120000 : 0.401610 vs. 0.411810
# Result for weight 0.130000 : 0.402057 vs. 0.411810
# Result for weight 0.140000 : 0.402117 vs. 0.411810
# Result for weight 0.150000 : 0.401787 vs. 0.411810
# Result for weight 0.160000 : 0.401979 vs. 0.411810
# Result for weight 0.170000 : 0.402207 vs. 0.411810
# Result for weight 0.180000 : 0.402426 vs. 0.411810
# Result for weight 0.190000 : 0.403447 vs. 0.411810
# Result for weight 0.200000 : 0.403533 vs. 0.411810

# testweight, testgrid = totVarApplication(0.12, 0.15, 0.4, testScenes, testLoss);
# Result for weight 0.120000 : 0.401912 vs. 0.413084