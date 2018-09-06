from numpy import *
import matplotlib.pyplot as plt

# load the dataset
def loadDataset(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    # print(numFeat)
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1])) # get the last number as label
    # print(dataMat[0:2])
    return dataMat, labelMat

def standRegres(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws

# get ws, weights, which we multiply by our constant tern
xArr, yArr = loadDataset('ex0.txt')
ws = standRegres(xArr, yArr)
print(ws)

xMat = mat(xArr); yMat = mat(yArr)
yHat = xMat * ws

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:, 0].flatten().A[0])

plt.show()


