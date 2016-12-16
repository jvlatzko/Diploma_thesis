#!/usr/local/bin/python3.5

import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float, io

import os
import sys  
cwd = os.getcwd()

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

def Lorentzian(Img, s = 50):
    return np.log(1.0 + s * Img**2)

def LorentzianLoss(Img, Ground, s = 50): 
    tImg = Img - Ground; 
    tImg -= np.mean(tImg); 
    return np.mean(np.log(1.0 + s*(tImg)**2))

def readResults(rtfexp_name, scene_name, conn, depth):
    groundTruth = np.loadtxt(dataPath + "labels/" + scene_name + ".dlm", delimiter='\t') # Get GT from data folder
    # GT DONE
    allInputRTF = np.loadtxt(dataPath + "images/" + scene_name + ".dlm", delimiter='\t') # Get INPT from data folder
    inputRTF = allInputRTF[:, 8:1800:9]
    # INPUT DONE
    outputRTF = np.loadtxt("%s/%s RTFTrainedForMyLorentzian_%dx%d_%d.dlm"%(rtfexp_name, scene_name, conn, conn, depth), delimiter='\t')
    #OUTPUT DONE    
    return groundTruth, inputRTF, outputRTF
    
def writeStats(RTF_experiment, conn, depth, setType, S): 
    nameStr = "%s/Evaluation_Results_RTFTrainedForMyLorentzian_%dx%d_%d_%s.txt"%(RTF_experiment, conn, conn, depth, setType)
    np.savetxt(nameStr, S, delimiter='\t', fmt='%.8f'); 

def evaluateScenes(rtfexp_name, sceneSet, conn, depth): 
    totalLoss = []; totalFormLoss = []; 
    print("Number of scenes: %d"%len(sceneSet))
    S = np.zeros((2 + len(sceneSet),5)) 
    ii = 2; 
    for scene in sceneSet:
        # print("In scene %s:"%scene)
        GT, iRTF, oRTF = readResults(rtfexp_name, scene, conn, depth); 
        diffIm1 = oRTF - GT; 
        
        currLoss = LorentzianLoss(GT, oRTF); 
        formLoss = LorentzianLoss(GT, iRTF);
        
        totalLoss.append(currLoss); 
        totalFormLoss.append(formLoss); 
        S[ii, :] = [np.min(oRTF), np.max(oRTF), np.mean(diffIm1), formLoss, currLoss]
        saveScene(scene, True)
        ii += 1; 
        #S.append((np.min(oRTF), np.max(oRTF), np.mean(diffIm1), formLoss, currLoss))
    # writeStats(rtfexp_name, conn, depth, sceneSet)
    print("%s total loss: before @ %.5f, after @%.5f"%(rtfexp_name, np.mean(totalFormLoss), np.mean(totalLoss)))
    S[0, 0] = np.mean(totalFormLoss); S[0,1] = np.mean(totalLoss);
    return np.mean(totalLoss), np.mean(totalFormLoss), S


def visualizeScene(rtfexp_name, sceneName, conn, depth, deltaSwitch = True):
    GT, iRTF, oRTF = readResults(rtfexp_name, sceneName, conn, depth); 
    vmin, vmax = np.min(GT), np.max(GT); 
    plt.figure(figsize=(14,3)); 
    plt.subplot(131); plt.imshow(iRTF, interpolation='nearest', vmin=vmin, vmax=vmax); plt.title("Input %s"%LorentzianLoss(iRTF, GT)); plt.colorbar(); 
    plt.subplot(132); plt.imshow(oRTF, interpolation='nearest', vmin=vmin, vmax=vmax); plt.title("Output %s"%LorentzianLoss(oRTF, GT)); plt.colorbar(); 
    plt.subplot(133); plt.imshow(GT, interpolation='nearest', vmin=vmin, vmax=vmax); plt.title("Ground Truth %s"%sceneName); plt.colorbar(); 
    if( deltaSwitch): 
        plt.figure(figsize=(14,3)); 
        deltaIn = iRTF - GT; deltaIn -= np.mean(deltaIn); deltaIn = Lorentzian(deltaIn); 
        deltaOut = oRTF - GT; deltaOut -= np.mean(deltaOut); deltaOut = Lorentzian(deltaOut); 
        improv = deltaOut - deltaIn; vmindel = np.min(improv); vmaxdel = np.max(improv); 
        vmin, vmax = np.min((np.min(deltaIn), np.min(deltaOut))), np.max((np.max(deltaIn), np.max(deltaOut))); 
        plt.subplot(131); plt.imshow(deltaIn, interpolation='nearest', vmin=vmin, vmax=vmax); plt.title("Delta Input"); plt.colorbar(); 
        plt.subplot(132); plt.imshow(deltaOut, interpolation='nearest', vmin=vmin, vmax=vmax); plt.title("Delta Output"); plt.colorbar();
        plt.subplot(133); plt.imshow(improv, interpolation='nearest', vmin = vmindel, vmax = vmaxdel); plt.title("Difference"); plt.colorbar(); 
    plt.show() 
    
def saveScene(sceneName, deltaSwitch = True):
    GT, iRTF, oRTF = readResults(rtfexp_name, scene, conn, depth); 
    vmin, vmax = np.min(gt), np.max(gt); 
    plt.figure(figsize=(14,3)); 
    plt.subplot(131); plt.imshow(iRTF, interpolation='nearest', vmin=vmin, vmax=vmax); plt.axis('off'); plt.title("Input %s"%float('%.5g' % LorentzianLoss(iRTF, GT)) ); plt.colorbar(); 
    plt.subplot(132); plt.imshow(oRTF, interpolation='nearest', vmin=vmin, vmax=vmax); plt.axis('off'); plt.title("Output %s"%float('%.5g' % LorentzianLoss(oRTF, GT)) ); plt.colorbar(); 
    plt.subplot(133); plt.imshow(GT, interpolation='nearest', vmin=vmin, vmax=vmax); plt.axis('off'); plt.title("Ground Truth %s"%sceneName); plt.colorbar(); 
    plt.savefig('%s_RTF.png'%sceneName, bbox_inches='tight')
    if( deltaSwitch): 
        plt.figure(figsize=(14,3)); plt.axis('off')
        deltaIn = iRTF - GT; deltaIn -= np.mean(deltaIn); deltaIn = Lorentzian(deltaIn); 
        deltaOut = oRTF - GT; deltaOut -= np.mean(deltaOut); deltaOut = Lorentzian(deltaOut); 
        improv = deltaOut - deltaIn; vmindel = np.min(improv); vmaxdel = np.max(improv); 
        vmin, vmax = np.min((np.min(deltaIn), np.min(deltaOut))), np.max((np.max(deltaIn), np.max(deltaOut))); 
        plt.subplot(131); plt.imshow(deltaIn, interpolation='nearest', vmin=vmin, vmax=vmax); plt.axis('off'); plt.title("Delta Input"); plt.colorbar(); 
        plt.subplot(132); plt.imshow(deltaOut, interpolation='nearest', vmin=vmin, vmax=vmax); plt.axis('off'); plt.title("Delta Output"); plt.colorbar();
        plt.subplot(133); plt.imshow(improv, interpolation='nearest', vmin = vmindel, vmax = vmaxdel); plt.axis('off'); plt.title("Difference"); plt.colorbar(); 
        plt.savefig('%s_RTF_delta.png'%sceneName, bbox_inches='tight')

def getVal(default):
    r = input().strip()
    if r == '':
        return default
    else:
        return r

print(cwd)
if __name__ == "__main__":
    print("Please enter path to data: ")
    dataPath = getVal("../Train_Val_Set/")
#RTFOutputPath = "3x3_6_lambda50_E2/"
#trainedType = " RTFTrainedForMyLorentzian_1x1_7" # note the leading space

    trainScenes = getScenes("Train")
    testScenes  = getScenes("Test")
    RTF_experiment, RTF_connectivity, RTF_depth = "1x1_10_lambda50_E2", 1, 10; # and so on
    print("Enter RTF experiment name, connectivity and depth (separate by return)")
    RTF_experiment, RTF_connectivity, RTF_depth = getVal("1x1_10_lambda50_E2"), int(getVal("1")), int(getVal("10")); 
    
    trainTotal_RTF_loss, trainTotal_formula_loss, Strain = evaluateScenes(RTF_experiment, trainScenes, RTF_connectivity, RTF_depth)
    writeStats(RTF_experiment, RTF_connectivity, RTF_depth, "train", Strain)
    testTotal_RTF_loss, testTotal_formula_loss, Stest = evaluateScenes(RTF_experiment, testScenes, RTF_connectivity, RTF_depth)
    writeStats(RTF_experiment, RTF_connectivity, RTF_depth, "test", Stest)
    
    shuffled_scenes = testScenes.copy();
    np.random.shuffle(testScenes); 
    print("How many random scenes should be chosen for display? ")
    n = int(getVal(4))
    plt.viridis()
    for scene in np.random.permutation(testScenes)[0:n]:
        visualizeScene(RTF_experiment, scene, RTF_connectivity, RTF_depth, True);   