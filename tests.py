from NNFullyCon import NNFullyCon as NNFC
from NNFullyConTrainer import NNFullyConTrainer as NNFCT
import GeneticNN
import numpy as np
import unittest

class basicTest(unittest.TestCase):

    def test_basic(self):
        net = NNFC([2,1,2])

        numOfExamples = 10000
        sizeOfSet = 2
        intervalMax = 100000
        # trainBatch = np.random.rand(numOfExamples, sizeOfSet) * 100
        trainBatch = np.random.choice(intervalMax, size=(numOfExamples, sizeOfSet), replace=False).astype(
            float) / intervalMax
        yTrainBatch = np.argmin(trainBatch, axis=1)

        # testBatch = np.random.rand(numOfExamples, sizeOfSet) * 100
        testBatch = np.random.choice(intervalMax, size=(numOfExamples, sizeOfSet), replace=False).astype(
            float) / intervalMax
        yTest = np.argmin(testBatch, axis=1)

        acc = net.score(testBatch, yTest)

        print ("Accuracy: " + str(acc))

    def test_changeWeights(self):
        net = NNFC([2, 1, 2])

        numOfExamples = 10000
        sizeOfSet = 2
        intervalMax = 100000

        # testBatch = np.random.rand(numOfExamples, sizeOfSet) * 100
        testBatch = np.random.choice(intervalMax, size=(numOfExamples, sizeOfSet), replace=False).astype(
            float) / intervalMax
        yTest = np.argmin(testBatch, axis=1)

        acc = net.score(testBatch, yTest)
        print ("Accuracy: " + str(acc))

        wList = net.getParametersList()
        newParams = [np.zeros(param.shape) for param in wList]
        net.updateParameters(newParams)

        acc2 = net.score(testBatch, yTest)
        print ("Accuracy: " + str(acc2))

    def test_getSubInterval(self):
        interval = GeneticNN.SearchInterval(-9, 9)
        gnn = GeneticNN.GeneticNN(interval)

        sub1 = gnn.getSubInterval(interval, subIntervalIdx=0)
        self.assertEqual(sub1, GeneticNN.SearchInterval(-9, -9+(18/3) ))

        sub2 = gnn.getSubInterval(interval, 1)
        self.assertEqual(sub2, GeneticNN.SearchInterval(-9 + (18 / 3), -9 + 2*(18 / 3)))

        #   test sub interval:
        subinterval = GeneticNN.SearchInterval(-3,3)
        sub1 = gnn.getSubInterval(subinterval, 0)
        self.assertEqual(sub1, GeneticNN.SearchInterval(-3 + 0*(6 / 3), -3 + 1*(6 / 3)) )

        sub2 = gnn.getSubInterval(subinterval, 1)
        self.assertEqual(sub2, GeneticNN.SearchInterval(-3 + (6 / 3), -3 + 2 * (6 / 3)))

        sub3 = gnn.getSubInterval(subinterval, 2)
        self.assertEqual(sub3, GeneticNN.SearchInterval(-3 + 2*(6 / 3), -3 + 3 * (6 / 3)))

        #   test lower subinterval:
        subinterval = GeneticNN.SearchInterval(-9, -3)
        sub1 = gnn.getSubInterval(subinterval, 0)
        self.assertEqual(sub1, GeneticNN.SearchInterval(-9 + 0 * (6 / 3), -9 + 1 * (6 / 3)))

        sub2 = gnn.getSubInterval(subinterval, 1)
        self.assertEqual(sub2, GeneticNN.SearchInterval(-9 + (6 / 3), -9 + 2 * (6 / 3)))

        sub3 = gnn.getSubInterval(subinterval, 2)
        self.assertEqual(sub3, GeneticNN.SearchInterval(-9 + 2 * (6 / 3), -9 + 3 * (6 / 3)))

    # def test_paramTuning(self):
    #     net = NNFC([2, 1, 2])
    #
    #     numOfExamples = 10000
    #     sizeOfSet = 2
    #     intervalMax = 100000
    #
    #     # testBatch = np.random.rand(numOfExamples, sizeOfSet) * 100
    #     testBatch = np.random.choice(intervalMax, size=(numOfExamples, sizeOfSet), replace=False).astype(
    #         float) / intervalMax
    #     yTest = np.argmin(testBatch, axis=1)
    #
    #     acc = net.score(testBatch, yTest)
    #     print ("Accuracy: " + str(acc))
    #
    #     gnn = GeneticNN.GeneticNN([-10, 10])
    #     gnn.tuneParameters(net, testBatch, yTest)
    #
    #     acc2 = net.score(testBatch, yTest)
    #     print ("Accuracy: " + str(acc2))


if __name__ == '__main__':
    # unittest.main()
    netTrainer = NNFCT([10, 100, 100, 10])

    numOfExamples = 10000
    sizeOfSet = 10
    intervalMax = 100000

    # testBatch = np.random.rand(numOfExamples, sizeOfSet) * 100
    testBatch = np.random.choice(intervalMax, size=(numOfExamples, sizeOfSet), replace=False).astype(
        float) / intervalMax
    yTest = np.argmin(testBatch, axis=1)

    acc = netTrainer.score(testBatch, yTest)
    print ("Accuracy: " + str(acc))

    import time
    timestamp = int(time.time())

    gnn = GeneticNN.GeneticNN(GeneticNN.SearchInterval(-1.5, 1.5), logFile="logs/log{}.txt".format(timestamp))
    gnn.tuneParameters(netTrainer, testBatch, yTest)

    acc2 = netTrainer.score(testBatch, yTest)
    print ("Accuracy: " + str(acc2))
