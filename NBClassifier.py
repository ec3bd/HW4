#Eamon Collins      ec3bd
#Caroline Holmes    cbh4ct

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
    augmented = concatenate((xVal,yVal),1)
    random.seed(37)
    random.shuffle(augmented)
    augmented = asmatrix(augmented)
    xVal = asmatrix(augmented[:, 0:p])
    yVal = asmatrix(augmented[:, p])
    #print(xVal[9])
    #print(yVal)
    #thetas, falseThetas, numClass = train(xVal, yVal)
    #Accuracy = test(thetas, falseThetas, xVal, yVal, numClass)
    #print("train/test: " + repr(Accuracy))
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

        thetas, falseThetas, numClass = train(xTrain, yTrain)
        Accuracy = test(thetas, falseThetas, xTest, yTest, numClass)
        print(repr(i) + ": " + repr(Accuracy))

#trains on the labeled samples passed in
#returns dict, each class name maps to the list of probabilities for that class
def train(xTrain, yTrain):
    n = xTrain.shape[0]
    p = xTrain.shape[1]
    keys = ['brazilian', 'british', 'cajun_creole','chinese','filipino','french','greek','indian','irish','italian','jamaican','japanese','korean','mexican','moroccan','russian','southern_us','spanish','thai','vietnamese']
    value = [0 for x in range(p)]
    thetas = {key: list(value) for key in keys}
    falseThetas = {key: list(value) for key in keys}
    numClass = {key: 0 for key in keys}
    for i in range(n):
        numClass[yTrain[i,0]] += 1
        for j in range(p):
            thetas[yTrain[i,0]][j] += int(xTrain[i,j])
            falseThetas[yTrain[i,0]][j] += 1 - int(xTrain[i,j])
    print(thetas['italian'])
    print(thetas['brazilian'])
    #print(numClass)
    sums = dict()
    for key in thetas:
        sums[key] = sum(thetas[key])
    for key in thetas:
        for j in range(p):
            falseThetas[key][j] = (falseThetas[key][j] + 1) / (numClass[key] + 4)
            thetas[key][j] = (thetas[key][j] + 1) / (numClass[key] + 4)#word doc results using sums[key] + p
    #print(thetas['greek'])
    print(thetas['italian'])
    print(falseThetas['italian'])
    print(thetas['brazilian'])
    return thetas, falseThetas, numClass

#Makes a prediction about the samples and compares it to the label for that sample
#returns an accuracy value
def test(thetas, falseThetas, xTest, yTest, numClass):
    n = xTest.shape[0]
    p = xTest.shape[1]
    if n == 1794:
        numTrain = 1794
    else:
        numTrain = 1794 - n
    yPredict = []
    keys = ['brazilian', 'british', 'cajun_creole','chinese','filipino','french','greek','indian','irish','italian','jamaican','japanese','korean','mexican','moroccan','russian','southern_us','spanish','thai','vietnamese']
    value = [0 for x in range(n)]
    classChance = {key: list(value) for key in keys}
    for key in thetas:
        for i in range(n):
            classChance[key][i] += math.log(float(numClass[key])/numTrain) #p(c)
            for j in range(p):
                if int(xTest[i,j]) == 0:
                    classChance[key][i] += math.log(falseThetas[key][j])
                else:
                    classChance[key][i] += math.log(thetas[key][j])
    #print(thetas) very long and still all the same value at this stage
    print(classChance['italian'][0])
    print(classChance['italian'][3])
    print(classChance['brazilian'][0])
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