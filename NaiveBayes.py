import numpy as np
import math

def p(numberofOnes, numberofPoints):
    n = np.size(numberofOnes)
    probability = []
    for i in range(n):
        p = (numberofOnes[i]+1)/(numberofPoints+2)
        probability.append(p)
    return probability

def compare(probArrLabelOne, probArrLabelZero, plabelOne, plabelZero, testData):
    n = np.size(probArrLabelOne)
    multipleOne = 1;
    multipleZero = 1;
    for i in range(n):
        if(testData[i] > 0):
            multipleOne *= probArrLabelOne[i]
            multipleZero *= probArrLabelZero[i]
        else:
            multipleOne *= (1-probArrLabelOne[i])
            multipleZero *= (1-probArrLabelZero[i])
    multipleZero *= plabelZero
    multipleOne *= plabelOne
    return multipleOne, multipleZero


def errorRate(testLabels, labelDetect):
    m = np.size(testLabels)
    numberofError = 0
    for i in range(m):
        if(int(testLabels[i]) != labelDetect[i]):
            numberofError += 1

    return numberofError/m



def main():
    trainData = np.loadtxt(open("C:\\test\\SpectTrainData.csv", "rb"), delimiter= ",", skiprows=0)
    trainLabels = np.loadtxt(open("C:\\test\\SpectTrainLabels.csv", "rb"), delimiter= ",", skiprows=0)
    testData = np.loadtxt(open("C:\\test\\SpectTestData.csv", "rb"), delimiter= ",", skiprows=0)
    testLabels = np.loadtxt(open("C:\\test\\SpectTestLabels.csv", "rb"), delimiter= ",", skiprows=0)

    m, n = np.shape(trainData)
    labelOne = np.sum(trainLabels,axis=0)
    labelZero = m - labelOne
    plabelOne = labelOne / m
    plabelZero = 1 - plabelOne

    indexMaxtoMin = np.argsort(-trainLabels)
    trainDatawithLabelOne = trainData[indexMaxtoMin[:int(labelOne)]]
    numberofOnesofLabelOne = np.sum(trainDatawithLabelOne, axis=0)
    probArrLabelOne = p(numberofOnesofLabelOne,labelOne)


    trainDataWithLabelZero = trainData[indexMaxtoMin[int(labelOne): m]]
    numberofOnesofLabelZero = np.sum(trainDataWithLabelZero, axis=0)
    probArrLabelZero = p(numberofOnesofLabelZero, labelZero)


    labelDetect = []
    x, y = np.shape(testData)

    for i in range(x):
        pOne, pZero = compare(probArrLabelOne, probArrLabelZero, plabelOne, plabelZero, testData[i])

        if(pOne >= pZero):
            labelDetect.append(1)
        else:
            labelDetect.append(0)


    rateofError = errorRate(testLabels, labelDetect)
    return rateofError





if __name__ == "__main__":
    rate = main()

    print("The detect error rate is:")
    print(rate)
