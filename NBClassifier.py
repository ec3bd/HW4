#Eamon Collins      ec3bd
#Caroline Holmes    fffff

from numpy import *
import json


def main():
    xVal, yVal = loadData()
    cv(xVal, yVal)

def loadData():
    ingrInd = dict()
    with open("ingredients.json") as data:
        jsonObj = json.load(data)
    i = 0
    for ingr in jsonObj["ingredients"]:
        ingrInd.update(dict.fromkeys([ingr],i))
        i+=1
    xVal = []
    yVal = []
    file = open("training.csv")
    for line in file:
        line = line.split(",")
        ingrVect = [0 for x in range(len(ingrInd.keys()))]
        for j in range(2,len(line)):
            ingrVect[ingrInd[line[j]]] += 1
        xVal.append(ingrVect)
        yVal.append(line[1])
    xVal = asmatrix(xVal)
    yVal = asmatrix(yVal)
    return xVal, yVal

def cv(xVal, yVal):
    n = xVal.shape[0]
    p = xVal.shape[1]
    for i in range(0,):
        beg = int(i*n/10)
        end = int((i+1)*n/10)
        xTest = xVal[beg:end,:]
        xTrain = xVal[0:beg:1,:]
        xTrain2 = xVal[end:,:]
        xTrain = concatenate((xTrain, xTrain2))
        yTest = yVal[beg:end,:]
        yTrain = yVal[0:beg:1,:]
        yTrain2 = yVal[end:,:]
        yTrain = concatenate((yTrain, yTrain2))

        thetas = train(xTrain, yTrain)
        test(thetas, xTest, yTest)


def train(xTrain, yTrain):
    n = xTrain.shape[0]
    p = xTrain.shape[1]
    thetas = dict.fromkeys(['brazilian', 'british', 'cajun_creole','chinese','filipino','french','greek','indian','irish','italian','jamaican','japanese','korean','mexican','moroccan','russian','southern_us','spanish','thai','vietnamese'], [0 for x in range(p)])
    numClass = dict.fromkeys(['brazilian', 'british', 'cajun_creole','chinese','filipino','french','greek','indian','irish','italian','jamaican','japanese','korean','mexican','moroccan','russian','southern_us','spanish','thai','vietnamese'], [0 for x in range(p)])
    for i in range(n):
        numClass[yTrain[i]] += 1
        for j in range(p):
            thetas[yTrain[i]][p] += xTrain[i,p]

    sums = dict()
    for key in thetas:
        sums[key] = sum(thetas[key])
    for key in thetas:
        for j in range(p):
            thetas[key][j] = (thetas[key][j] + 1) / (sums[key] + numClass[key])
    return thetas

def test(thetas, xTest, yTest):
    n = xTest.shape[0]
    p = xTest.shape[1]
    
if __name__ == "__main__":
    main()