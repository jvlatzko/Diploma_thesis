#!/usr/local/bin/python3.5

import numpy as np
import matplotlib.pyplot as plt


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

def myMSE(Img, Ground):
    tImg = Img - Ground; 
    tImg -= np.mean(tImg); 
    return np.mean( np.sqrt(np.square(tImg)) )

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

def writeTestStats(RTF_experiment, conn, depth, S, setType = "trueTest", cascadeComparison = False): 
    nameStr = "%s_test/Evaluation_Results_RTF_%dx%d_%d_%s.txt"%(RTF_experiment, conn, conn, depth, setType)
    if( cascadeComparison): 
        nameStr = "%s_test/Evaluation_Results_RTF_%dx%d_%d_%s_cascade.txt"%(RTF_experiment, conn, conn, depth, setType)
    np.savetxt(nameStr, S, delimiter='\t', fmt='%.8f'); 

def evaluateScenes(rtfexp_name, conn, depth, sceneSet, cascadeComparison = True): 
    totalFormLoss = []; totalMSE = []; totalMSE_RTF = []; totalMSE_RTFc = []; totalLoss_RTF = []; totalLoss_RTFc = []; 
    print("Number of scenes: %d"%len(sceneSet))
    S = np.zeros((2 + len(sceneSet),6)) 
    ii = 2; 
    if (setType.find("rueT") > 0):
        testCase = 1;
    for scene in sceneSet:
        if (testCase): 
            GT, iRTF, oRTF = readTestResults(rtfexp_name, conn, depth, scene); 
        else:
            GT, iRTF, oRTF = readValResults(rtfexp_name, conn, depth, scene);
        formLoss = LorentzianLoss(GT, iRTF);
        totalFormLoss.append(formLoss); 
        err_MSE = myMSE(GT, iRTF)
        totalMSE.append(err_MSE); 
        loss_RTF = LorentzianLoss(GT, oRTF);
        totalLoss_RTF.append(loss_RTF); 
        err_RTF = myMSE(GT, oRTF); 
        totalMSE_RTF.append(err_RTF); 
        saveScene(rtfexp_name, conn, depth, scene, True)
        if( cascadeComparison ): 
            if (testCase): 
                GT, iRTF, oRTF = readTestResults(rtfexp_name, conn, depth, scene); 
            else:
                GT, iRTF, oRTF = readValResults(rtfexp_name, conn, depth, scene);  
            loss_RTFc = LorentzianLoss(GT, oRTFc); 
            totalLoss_RTFc.append(loss_RTFc); 
            err_RTFc = myMSE(GT, oRTFc)
            totalMSE_RTFc.append(err_RTFc); 
            S[ii, :] = [formLoss, err_MSE, loss_RTF,  err_RTF, loss_RTFc, err_RTFc]
            saveCascadeComparison(scene, rtfexp_name, conn, depth); 
        else: 
            S[ii, 0:4] = [formLoss, err_MSE, loss_RTF,  err_RTF]
        ii += 1; 
    print("%s total loss: before @ %.5f, after @ %.5f"%(rtfexp_name, np.mean(totalFormLoss), np.mean(totalLoss_RTF)))
    S[0, 0] = np.mean(totalFormLoss); S[0, 1] = np.mean(totalMSE); S[0, 2] = np.mean(totalLoss_RTF); S[0, 3] = np.mean(totalMSE_RTF)
    if(cascadeComparison): 
        S[0, 4] = np.mean(totalLoss_RTFc);
        S[0, 5] = np.mean(totalMSE_RTFc)
    return S

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
   # vmin, vmax = np.min(), np.max();
    plt.subplot(223); plt.imshow(oRTF, interpolation='nearest', vmin=vmin, vmax=vmax); plt.axis('off'); plt.title("Loss RTF (%s)"%float('%.5g' % LorentzianLoss(oRTF, GT)) ); plt.colorbar();
    plt.subplot(224); plt.imshow(oRTFc, interpolation='nearest', vmin=vmin, vmax=vmax); plt.axis('off'); plt.title("Loss Cascade (%s)"%float('%.5g' % LorentzianLoss(oRTFc, GT)) ); plt.colorbar(); 
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
   # vmin, vmax = np.min(), np.max();
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

   # trainScenes = getScenes("Train")
    testScenes  = getScenes("Test")
    trueTestScenes  = getScenes("trueTest")
   # RTF_experiment, RTF_connectivity, RTF_depth = "1x1_10_lambda50_E2", 1, 10; # and so on
    print("Enter RTF experiment name, connectivity and depth (separate by return) [3x3_13, 3, 13]")
    RTF_experiment, RTF_connectivity, RTF_depth = getVal("3x3_13"), int(getVal("3")), int(getVal("13")); 
    
    #Strain = evaluateScenes(RTF_experiment, RTF_connectivity, RTF_depth, trainScenes)
    #writeStats(RTF_experiment, RTF_connectivity, RTF_depth, "train", Strain)
    Stest = evaluateScenes(RTF_experiment, RTF_connectivity, RTF_depth, testScenes)
    #writeStats(RTF_experiment, RTF_connectivity, RTF_depth, "test", Stest)
   # StrueTest = evaluateScenes(RTF_experiment, RTF_connectivity, RTF_depth, trueTestScenes)
    #writeStats(RTF_experiment, RTF_connectivity, RTF_depth, "trueTest", StrueTest)
    
    print("How many random scenes should be chosen for display? [4]")
    n = int(getVal("4"))
    for scene in np.random.permutation(trueTestScenes)[0:int(n)]:
        visualizeScene(RTF_experiment, RTF_connectivity, RTF_depth, scene, True);   