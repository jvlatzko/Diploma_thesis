#!/usr/local/bin/python3.5

import numpy as np

import os
import re
cwd = os.getcwd()

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

def writeStats(S): 
    nameStr = dataPath + "Scenic_Loss.txt"; 
    np.savetxt(nameStr, S, delimiter='\t', fmt='%.8f'); 

def evaluateScenes(sceneSet): 
    totalFormLoss = []; 
    print("Number of scenes: %d"%len(sceneSet))
    S = np.zeros((5 + len(sceneSet),1)) 
    ii = 0; 
    for scene in sceneSet:
        # print("In scene %s:"%scene)
        GT, iRTF = readResults(scene); 
        formLoss = LorentzianLoss(GT, iRTF);
        totalFormLoss.append(formLoss); 
        S[ii, :] = [formLoss]
        ii += 1; 
    print("Total loss over all data: %.5f"%(np.mean(totalFormLoss)))
    S[ii+4, 0] = np.mean(totalFormLoss); 
    return np.mean(totalFormLoss), S

def readResults(scene_name):
    groundTruth = np.loadtxt(dataPath + "labels/" + scene_name + ".dlm", delimiter='\t') # Get GT from data folder
    # GT DONE
    allInputRTF = np.loadtxt(dataPath + "images/" + scene_name + ".dlm", delimiter='\t') # Get INPT from data folder
    inputRTF = allInputRTF[:, 8:1800:9]
    # INPUT DONE
    return groundTruth, inputRTF

def getVal(default): 
    r = input().strip();
    if( r == ''):
        return default
    else:
        return r


print("Please enter path to data: ")
dataPath = getVal("../depth_data/ToF-data/Train_Val_Set/")

scenes = getScenes("Train")
print("Got training")
testScenes  = getScenes("Test")
print("Got test")
scenes.extend(testScenes); scenes = sorted(scenes); 
print("Loaded scenes")
total_formula_loss, S = evaluateScenes(scenes)
print("Evaluated scenes")
writeStats(S)

sepScenes; 
pattern_bath = r"bath"; baths = []; 
pattern_bed = r"bed"; beds = []; 
pattern_allKit = r"kitchen"; allKit = [];
pattern_kit = r"kitchen_[0-1][0-9]"; kit = []; 
pattern_kit_e = r"easy"; kit_e = []; 
pattern_kit_r = r"right"; kit_r = []; 
pattern_kit_v = r"_v"; kit_v = []; 
pattern_sitt = r"sitting"; sitts = []; 

for i in range(len(scenes)):
    if( re.match(pattern_bath, scenes[i])):
        baths.append(scenes[i])
    elif( re.match(pattern_bed, scenes[i])):
        beds.append(scenes[i])
    elif( re.match(pattern_kit, scenes[i])):
        kit.append(scenes[i])
    elif( re.search(pattern_kit_e, scenes[i])):
        kit_e.append(scenes[i])
    elif( re.search(pattern_kit_r, scenes[i])):
        kit_r.append(scenes[i])
    elif( re.search(pattern_kit_v, scenes[i])):
        kit_v.append(scenes[i])
    elif( re.match(pattern_sitt, scenes[i])):
        sitts.append(scenes[i])
    if( re.match(pattern_allKit, scenes[i])):
        allKit.append(scenes[i])

loss_bath, S_bath = evaluateScenes(baths)
loss_bed, S_bed = evaluateScenes(beds)
loss_kit, S_kit = evaluateScenes(kit)
loss_kit_e, S_kite = evaluateScenes(kit_e)
loss_kit_r, S_kitr = evaluateScenes(kit_r)
loss_kit_v, S_kitv = evaluateScenes(kit_v)
loss_sitt, S_sitt = evaluateScenes(sitts)
loss_allKit, S_allKit = evaluateScenes(allKit); 

print("Done. ")