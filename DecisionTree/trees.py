from math import log
import operator

def calcShannonEnt(dataset):
    numEntries = len(dataset)
    labelCounts = {}
    for featVec in dataset:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels


myDat, labels = createDataSet()

# print(calcShannonEnt(myDat))
# print(-3/5*log(3/5, 2) - 2/5*log(2/5, 2))

def splitDataset(dataset, axis, value):
    retDataset = []
    for featVec in dataset:
        if featVec[axis] == value:
            print(featVec)
            reducedFeatVec = featVec[: axis]
            reducedFeatVec.extend(featVec[axis+1 :])
            retDataset.append(reducedFeatVec)
            print('-----begin----')
            print(featVec[: axis])
            print(featVec[axis+1 :])
            print('----end-------')
    print(retDataset)
    return retDataset

splitDataset(myDat, 1, 1)

def chooseBestFeatureToSplit(dataset):
    numFeatures = len(dataset[0]) - 1
    baseEntropy = calcShannonEnt(dataset)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataset]
        print(featList)
        uniqueVals = set(featList)
        print(uniqueVals)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataset = splitDataset(dataset, i, value)
            prob = len(subDataset)/float(len(dataset))
            newEntropy += prob * calcShannonEnt(subDataset)
        infoGain = baseEntropy - newEntropy
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

# chooseBestFeatureToSplit(myDat)

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),\
                        keys = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

def createTree(dataset, labels):
    classList = [example[-1] for example in dataset]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataset[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataset)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{} }
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataset]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataset\
                                        (dataset, bestFeat, value), subLabels)
    return myTree
