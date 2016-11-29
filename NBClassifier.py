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
    file = open("training.json",'r')
    while(True):
        line = file.readline()
        if line == "":
            break
        line = json.loads(line.strip())
        ingrVect = [0 for x in range(len(ingrInd.keys()))]
        for ingredient in line["ingredients"]:
            ingrVect[ingrInd[ingredient]] += 1
        xVal.append(ingrVect)
        yVal.append(line["cuisine"])
    file.close()
    xVal = asmatrix(xVal)
    yVal = asmatrix(yVal).T

    return xVal, yVal

#trains and tests the data using 6-fold cross validation
def cv(xVal, yVal):
    n = xVal.shape[0]
    p = xVal.shape[1]

    for i in range(0,6):
        beg = int(i*n/6)
        end = int((i+1)*n/6)
        xTest = xVal[beg:end,:]
        xTrain = xVal[0:beg:1,:]
        xTrain2 = xVal[end:,:]
        xTrain = concatenate((xTrain, xTrain2))
        yTest = yVal[beg:end,:]
        yTrain = yVal[0:beg:1,:]
        yTrain2 = yVal[end:,:]
        yTrain = concatenate((yTrain, yTrain2))

        thetas = train(xTrain, yTrain)
        Accuracy = test(thetas, xTest, yTest)
        print(repr(i) + ": " + repr(Accuracy))

#trains on the labeled samples passed in
#returns dict, each class name maps to the list of probabilities for that class
def train(xTrain, yTrain):
    n = xTrain.shape[0]
    p = xTrain.shape[1]
    keys = ['brazilian', 'british', 'cajun_creole','chinese','filipino','french','greek','indian','irish','italian','jamaican','japanese','korean','mexican','moroccan','russian','southern_us','spanish','thai','vietnamese']
    value = [0 for x in range(p)]
    thetas = {key: list(value) for key in keys}
    numClass = dict.fromkeys(['brazilian', 'british', 'cajun_creole','chinese','filipino','french','greek','indian','irish','italian','jamaican','japanese','korean','mexican','moroccan','russian','southern_us','spanish','thai','vietnamese'], 0)
    for i in range(n):
        numClass[yTrain[i,0]] += 1
        for j in range(p):
            thetas[yTrain[i,0]][j] += xTrain[i,j]
    print(thetas['vietnamese'])
    print(thetas['french'])
    sums = dict()
    for key in thetas:
        sums[key] = sum(thetas[key])
    for key in thetas:
        for j in range(p):
            thetas[key][j] = (thetas[key][j] + 1) / (2 + numClass[key])#maybe needs to be + len(thetas.keys())
    return thetas

#Makes a prediction about the samples and compares it to the label for that sample
#returns an accuracy value
def test(thetas, xTest, yTest):
    n = xTest.shape[0]
    p = xTest.shape[1]
    yPredict = []
    classChance = dict()
    for key in thetas:
        classChance[key] = [0 for i in range(n)]
    for key in thetas:
        for i in range(n):
            for j in range(p):
                chanceJ = math.log(thetas[key][j])
                if xTest[i,j] == 0:
                    chanceJ = math.log(1 - thetas[key][j])
                classChance[key][i] += chanceJ
    #print(thetas) very long and still all the same value at this stage
    print(classChance['indian'][0])
    print(classChance['greek'][0])
    for i in range(n):
        best = classChance['indian'][i]
        bestClass = 'indian'
        for key in thetas:
            if best < classChance[key][i]:
                best = classChance[key][i]
                bestClass = key
        yPredict.append(bestClass)
    print(yPredict)
    true = 0
    total = 0
    for i in range(n):
        if yPredict[i] == yTest[i,0]:
            true += 1
        total += 1
    Accuracy = true / total
    return Accuracy






if __name__ == "__main__":
    main()