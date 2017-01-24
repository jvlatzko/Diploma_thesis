import numpy as np
import random, re

def getAllScenes(dataPath = "./"):
    scenes = [line.strip() for line in open(dataPath + "allScenes.txt", 'r')]
    return scenes; 

def lorentzian(Img, s = 50):
    return np.log(1.0 + s * Img**2)

def lorentzianLoss(Img, Ground, s = 50): 
    tImg = Img - Ground; 
    tImg -= np.mean(tImg); 
    return np.mean(np.log(1.0 + s*(tImg)**2))

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
        
def shuffleScenes(sceneSet, N):
    # N = len(sceneSet)
    shuffledSubSet = random.sample(sceneSet, int(N)); 
    return shuffledSubSet

def separateSetsTrainValTest(sceneSet, trainP = 0.65, valP = 0.1, testP = 0.25):
    trainSet = []; valSet = []; testSet = []; 
    sceneTypes = {}
    sceneTypes["baths"]       = [elem for elem in allScenes if( re.search("bath", elem))]
    sceneTypes["beds"]        = [elem for elem in allScenes if( re.search("bed", elem))]
    sceneTypes["kitchens"]    = [elem for elem in allScenes if( re.search("kitchen", elem))]
    sceneTypes["sittings"]    = [elem for elem in allScenes if( re.search("sitting", elem))]
    for sType in sceneTypes:
        nScenes = len(sceneTypes[sType])
        shuffled = shuffleScenes(sceneTypes[sType], nScenes); 
        trainSet.append(shuffled[0:int(trainP*nScenes)])
        valSet.append(shuffled[int(trainP*nScenes):int((trainP + valP)*nScenes)])
        testSet.append(shuffled[int((trainP + valP)*nScenes):nScenes])
    flattenedTrainSet   = [elem for sublist in trainSet for elem in sublist]
    flattenedValSet     = [elem for sublist in valSet for elem in sublist]
    flattenedTestSet    = [elem for sublist in testSet for elem in sublist]
    return flattenedTrainSet, flattenedValSet, flattenedTestSet
    
def evaluateSetLoss(sceneSet, ddict): 
    tloss = 0.0; 
    for scene in sceneSet: 
        tloss += ddict[scene]
    return tloss/len(sceneSet)
    
def evaluateSeparation(trainSet, valSet, testSet, printy=False): 
    setLoss = {}
    for cname, cset in zip(["train", "val", "test"], [trainSet, valSet, testSet]): 
        setLoss[cname] = evaluateSetLoss(cset, losses)
        if(printy):
            print("Loss of %s:\t %f"%(cname, setLoss[cname]))
    return setLoss
        
def optimizeSeparation(sceneSet, ttol=0.05, printy=False):
    lossdiscr = ttol + 1.0; #dummy, assure execution
    numIts = 0
    while(lossdiscr > ttol):
        trainSet, valSet, testSet = separateSetsTrainValTest(sceneSet); 
        setLoss = evaluateSeparation(trainSet, valSet, testSet, printy); 
        trl, val, tel = setLoss["train"], setLoss["val"], setLoss["test"]
        lossdiscr = max(np.abs(trl-val), np.abs(val-tel), np.abs(trl-tel))
        numIts += 1; 
    print("Total of %d iterations for separation"%numIts)
    return trainSet, valSet, testSet
    
def writeSets(trainSet, valSet, testSet):
    with open("train.txt","w") as tr:
        tr.write("\n".join(sorted(trainSet)));
    with open("test.txt","w") as tr:
        va.write("\n".join(sorted(valSet)));
    with open("true_Test.txt","w") as tr:
        te.write("\n".join(sorted(testSet)));
    

allScenes = getAllScenes(); 
losses = storeLoss(allScenes) 
# iport time; t = time.time(); losses = storeLoss(allScenes); print("Done in %f s"%(time.time() - t))
# Done in 79.591195 s

trainSet, valSet, testSet = optimizeSeparation(allScenes, 0.002); 
# trainSet2, valSet2, testSet2 = optimizeSeparation(allScenes, 0.002);
# Total of 1192 iterations for separation
# trainSet, valSet, testSet = optimizeSeparation(allScenes, 0.01);
# Total of 76 iterations for separation

writeSets(trainSet, valSet, testSet); 