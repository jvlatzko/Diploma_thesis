#!/usr/local/bin/python3.5

import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float, io


def getScenes(fname):
    if (fname.find("rain") > 0):
        scenes = [line.strip() for line in open(dataPath + "train.txt", 'r')]
    elif (fname.find("rueT") > 0):
        scenes = [line.strip() for line in open(dataPath + "trueTest.txt", 'r')]
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

def readResults(rtfexp_name, conn, depth, scene_name):
    groundTruth = np.loadtxt(dataPath + "labels/" + scene_name + ".dlm", delimiter='\t') # Get GT from data folder
    # GT DONE
    allInputRTF = np.loadtxt(dataPath + "images/" + scene_name + ".dlm", delimiter='\t') # Get INPT from data folder
    inputRTF = allInputRTF[:, 8:1800:9]
    # INPUT DONE
    outputRTF = np.loadtxt("%s/%s RTFTrainedForMyLorentzian_%dx%d_%d.dlm"%(rtfexp_name, scene_name, conn, conn, depth), delimiter='\t')
    #OUTPUT DONE    
    return groundTruth, inputRTF, outputRTF

def readValResults(rtfexp_name, conn, depth, scene_name):
    groundTruth = np.loadtxt("%s/%s Ground.dlm"%(rtfexp_name, scene_name), delimiter='\t')# Get GT from data folder
    # GT DONE
    allInputRTF = np.loadtxt("%s/%s Input.dlm"%(rtfexp_name, scene_name), delimiter='\t') # Get INPT from data folder
    inputRTF = allInputRTF[:, 8:1800:9]
    # INPUT DONE
    outputRTF = np.loadtxt("%s/%s Prediction.dlm"%(rtfexp_name, scene_name), delimiter='\t')
    #OUTPUT DONE    
    return groundTruth, inputRTF, outputRTF


def readTestResults(rtfexp_name, conn, depth, scene_name):
    groundTruth = np.loadtxt("%s_test/%s Ground.dlm"%(rtfexp_name, scene_name), delimiter='\t')# Get GT from data folder
    # GT DONE
    allInputRTF = np.loadtxt("%s_test/%s Input.dlm"%(rtfexp_name, scene_name), delimiter='\t') # Get INPT from data folder
    inputRTF = allInputRTF[:, 8:1800:9]
    # INPUT DONE
    outputRTF = np.loadtxt("%s_test/%s Prediction.dlm"%(rtfexp_name, scene_name), delimiter='\t')
    #OUTPUT DONE    
    return groundTruth, inputRTF, outputRTF
    
def writeStats(RTF_experiment, conn, depth, setType, S): 
    nameStr = "%s/Evaluation_Results_RTFTrainedForMyLorentzian_%dx%d_%d_%s.txt"%(RTF_experiment, conn, conn, depth, setType)
    np.savetxt(nameStr, S, delimiter='\t', fmt='%.8f'); 

def evaluateScenes(rtfexp_name, conn, depth, sceneSet, cascadeComparison = True): 
    totalLoss = []; totalFormLoss = []; 
    print("Number of scenes: %d"%len(sceneSet))
    S = np.zeros((2 + len(sceneSet),5)) 
    ii = 2; 
    for scene in sceneSet:
        GT, iRTF, oRTF = readValResults(rtfexp_name, conn, depth, scene); 
        diffIm1 = oRTF - oRTFc; 
        currLoss = LorentzianLoss(GT, oRTF); 
        currLossC = LorentzianLoss(GT, oRTFc); 
        formLoss = LorentzianLoss(GT, iRTF);
        totalLoss.append(currLoss); 
        totalFormLoss.append(formLoss); 
        S[ii, :] = [np.min(oRTF), np.max(oRTF), np.mean(diffIm1), formLoss, currLoss]
        saveScene(rtfexp_name, conn, depth, scene, True)
        if( cascadeComparison ): 
            GTc, iRTFc, oRTFc = readTestResults("%s_2"%rtfexp_name, conn, depth, scene); #Naming convention for CRTF
            saveCascadeComparison(scene, rtfexp_name, conn, depth); 
        ii += 1; 
        #S.append((np.min(oRTF), np.max(oRTF), np.mean(diffIm1), formLoss, currLoss))
    # writeStats(rtfexp_name, conn, depth, sceneSet)
    print("%s total loss: before @ %.5f, after @ %.5f"%(rtfexp_name, np.mean(totalFormLoss), np.mean(totalLoss)))
    S[0, 0] = np.mean(totalFormLoss); S[0,1] = np.mean(totalLoss);
    return np.mean(totalLoss), np.mean(totalFormLoss), S


def visualizeScene(rtfexp_name, conn, depth, sceneName, deltaSwitch = True):
    GT, iRTF, oRTF = readTestResults(rtfexp_name, conn, depth, sceneName); 
    vmin, vmax = np.min(GT), np.max(GT); 
    plt.figure(figsize=(14,3)); 
    plt.subplot(131); plt.imshow(iRTF, interpolation='nearest', vmin=vmin, vmax=vmax); plt.title("Baseline %s"%float('%.5g' %LorentzianLoss(iRTF, GT)) ); plt.colorbar(); plt.axis('off')
    plt.subplot(132); plt.imshow(oRTF, interpolation='nearest', vmin=vmin, vmax=vmax); plt.title("Prediction RTF %s"%float('%.5g' % LorentzianLoss(oRTF, GT)) ); plt.colorbar(); plt.axis('off')
    plt.subplot(133); plt.imshow(GT, interpolation='nearest', vmin=vmin, vmax=vmax); plt.title("Ground Truth %s"%sceneName); plt.colorbar(); plt.axis('off')
    if( deltaSwitch): 
        plt.figure(figsize=(14,3)); 
        deltaIn = iRTF - GT; deltaIn -= np.mean(deltaIn); deltaIn = Lorentzian(deltaIn); 
        deltaOut = oRTF - GT; deltaOut -= np.mean(deltaOut); deltaOut = Lorentzian(deltaOut); 
        improv = deltaOut - deltaIn; vmindel = np.min(improv); vmaxdel = np.max(improv); 
        vmin, vmax = np.min((np.min(deltaIn), np.min(deltaOut))), np.max((np.max(deltaIn), np.max(deltaOut))); 
        plt.subplot(131); plt.imshow(deltaIn, interpolation='nearest', vmin=vmin, vmax=vmax); plt.title("Loss Baseline"); plt.colorbar(); plt.axis('off')
        plt.subplot(132); plt.imshow(deltaOut, interpolation='nearest', vmin=vmin, vmax=vmax); plt.title("Loss Prediction"); plt.colorbar(); plt.axis('off')
        plt.subplot(133); plt.imshow(improv, interpolation='nearest', vmin = vmindel, vmax = vmaxdel); plt.title("Difference"); plt.colorbar(); plt.axis('off')
    plt.show() 
    
def saveScene(rtfexp_name, conn, depth, sceneName, deltaSwitch = True):
    GT, iRTF, oRTF = readTestResults(rtfexp_name, conn, depth, sceneName); 
    vmin, vmax = np.min(GT), np.max(GT); 
    plt.figure(figsize=(14,3)); 
    plt.subplot(131); plt.imshow(iRTF, interpolation='nearest', vmin=vmin, vmax=vmax); plt.axis('off'); plt.title("Baseline (%s)"%float('%.5g' % LorentzianLoss(iRTF, GT)) ); plt.colorbar(); 
    plt.subplot(132); plt.imshow(oRTF, interpolation='nearest', vmin=vmin, vmax=vmax); plt.axis('off'); plt.title("Prediction RTF (%s)"%float('%.5g' % LorentzianLoss(oRTF, GT)) ); plt.colorbar(); 
    plt.subplot(133); plt.imshow(GT, interpolation='nearest', vmin=vmin, vmax=vmax); plt.axis('off'); plt.title("Ground Truth %s"%sceneName); plt.colorbar(); 
    plt.savefig('%s_RTF.png'%sceneName, bbox_inches='tight')
    if( deltaSwitch): 
        plt.figure(figsize=(14,3)); plt.axis('off')
        deltaIn = iRTF - GT; deltaIn -= np.mean(deltaIn); deltaIn = Lorentzian(deltaIn); 
        deltaOut = oRTF - GT; deltaOut -= np.mean(deltaOut); deltaOut = Lorentzian(deltaOut); 
        improv = deltaOut - deltaIn; vmindel = np.min(improv); vmaxdel = np.max(improv); 
        vmin, vmax = np.min((np.min(deltaIn), np.min(deltaOut))), np.max((np.max(deltaIn), np.max(deltaOut))); 
        plt.subplot(131); plt.imshow(deltaIn, interpolation='nearest', vmin=vmin, vmax=vmax); plt.axis('off'); plt.title("Loss Baseline"); plt.colorbar(); 
        plt.subplot(132); plt.imshow(deltaOut, interpolation='nearest', vmin=vmin, vmax=vmax); plt.axis('off'); plt.title("Loss Prediction"); plt.colorbar();
        plt.subplot(133); plt.imshow(improv, interpolation='nearest', vmin = vmindel, vmax = vmaxdel); plt.axis('off'); plt.title("Difference"); plt.colorbar(); 
        plt.savefig('%s_RTF_delta.png'%sceneName, bbox_inches='tight')
        plt.close()
    plt.close()
    
def cascadeComparison(sceneName, rtfexp_name, conn, depth):
    GT, iRTF, oRTF = readTestResults(rtfexp_name, conn, depth, sceneName); 
    GTc, iRTFc, oRTFc = readTestResults(r"%s_2"%rtfexp_name, conn, depth, sceneName);
    diffIm1 = oRTF - oRTFc; 
    plt.figure(figsize=(14,12)); 
    vmin, vmax = np.min(GT), np.max(GT); 
    plt.subplot(221); plt.imshow(iRTF, interpolation='nearest', vmin=vmin, vmax=vmax); plt.axis('off'); plt.title("Baseline (%s)"%float('%.5g' % LorentzianLoss(iRTF, GT)) ); plt.colorbar(); 
    plt.subplot(222); plt.imshow(GT, interpolation='nearest', vmin=vmin, vmax=vmax); plt.axis('off'); plt.title("Ground Truth %s"%sceneName); plt.colorbar(); 
#    vmin, vmax = np.min(), np.max(); 
    plt.subplot(223); plt.imshow(oRTF, interpolation='nearest', vmin=vmin, vmax=vmax); plt.axis('off'); plt.title("Loss RTF (%s)"%float('%.5g' % LorentzianLoss(oRTF, GT)) ); plt.colorbar();
    plt.subplot(224); plt.imshow(oRTFc, interpolation='nearest', vmin=vmin, vmax=vmax); plt.axis('off'); plt.title("Loss Cascade %s"%float('%.5g' % LorentzianLoss(oRTFc, GT)) ); plt.colorbar(); 
    plt.show()
    #plt.savefig('%s_comparison.png'%sceneName, bbox_inches='tight')
    plt.close()
    
def saveCascadeComparison(sceneName, rtfexp_name, conn, depth):
    GT, iRTF, oRTF = readTestResults(rtfexp_name, conn, depth, sceneName); 
    GTc, iRTFc, oRTFc = readTestResults(r"%s_2"%rtfexp_name, conn, depth, sceneName);
    diffIm1 = oRTF - oRTFc; 
    plt.figure(figsize=(14,10)); 
    vmin, vmax = np.min(GT), np.max(GT); 
    plt.subplot(221); plt.imshow(iRTF, interpolation='nearest', vmin=vmin, vmax=vmax); plt.axis('off'); plt.title("Baseline (%s)"%float('%.5g' % LorentzianLoss(iRTF, GT)) ); plt.colorbar(); 
    plt.subplot(222); plt.imshow(GT, interpolation='nearest', vmin=vmin, vmax=vmax); plt.axis('off'); plt.title("Ground Truth %s"%sceneName); plt.colorbar(); 
#    vmin, vmax = np.min(), np.max(); 
    plt.subplot(223); plt.imshow(oRTF, interpolation='nearest', vmin=vmin, vmax=vmax); plt.axis('off'); plt.title("Loss RTF (%s)"%float('%.5g' % LorentzianLoss(oRTF, GT)) ); plt.colorbar(); 
    plt.subplot(224); plt.imshow(oRTFc, interpolation='nearest', vmin=vmin, vmax=vmax); plt.axis('off'); plt.title("Loss Cascade (%s)"%float('%.5g' % LorentzianLoss(oRTFc, GT)) ); plt.colorbar(); 
    #plt.show()
    plt.savefig('%s_comparison.png'%sceneName, bbox_inches='tight')
    plt.close()

def getVal(default):
    r = input().strip()
    if r == '':
        return default
    else:
        return r

print(cwd)
if __name__ == "__main__":
    plt.viridis()
    print("Please enter path to data: [../wholeset/]")
    dataPath = getVal("../wholeset/")

#    trainScenes = getScenes("Train")
    testScenes  = getScenes("Test")
    trueTestScenes  = getScenes("trueTest")
#    RTF_experiment, RTF_connectivity, RTF_depth = "1x1_10_lambda50_E2", 1, 10; # and so on
    print("Enter RTF experiment name, connectivity and depth (separate by return) [3x3_13, 3, 13]")
    RTF_experiment, RTF_connectivity, RTF_depth = getVal("3x3_13"), int(getVal("3")), int(getVal("13")); 
    
    #trainTotal_RTF_loss, trainTotal_formula_loss, Strain = evaluateScenes(RTF_experiment, RTF_connectivity, RTF_depth, trainScenes)
    #writeStats(RTF_experiment, RTF_connectivity, RTF_depth, "train", Strain)
    testTotal_RTF_loss, testTotal_formula_loss, Stest = evaluateScenes(RTF_experiment, RTF_connectivity, RTF_depth, testScenes)
    #writeStats(RTF_experiment, RTF_connectivity, RTF_depth, "test", Stest)
#    trueTestTotal_RTF_loss, trueTestTotal_formula_loss, StrueTest = evaluateScenes(RTF_experiment, RTF_connectivity, RTF_depth, trueTestScenes)
    #writeStats(RTF_experiment, RTF_connectivity, RTF_depth, "trueTest", StrueTest)
    
    print("How many random scenes should be chosen for display? [4]")
    n = int(getVal("4"))
    for scene in np.random.permutation(trueTestScenes)[0:int(n)]:
        visualizeScene(RTF_experiment, RTF_connectivity, RTF_depth, scene, True);   