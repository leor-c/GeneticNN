import tensorflow as tf
import numpy as np

class NNFullyConTrainer:
    """

    """
    def __init__(self, networkShape):
        """
        Constructs a fully connected NN with given shape. The shape includes input & out dimensions with at least 1
        hidden layer.
        :param networkShape: a tuple/list/nparray with at least 3 dimensions.
        """
        self.networkShape = networkShape
        self.parametersInfoList = None
        self.inputLayer = None
        self.logits = None
        self.graph = tf.Graph()

        self.feedDict = {}
        with self.graph.as_default():
            self.yHat = self.buildNetwork()
            self.prediction = self.getPrediction(self.yHat)
            self.sess = tf.Session()
            self.getParametersInfoList()
            self.labels = tf.placeholder(tf.int32, [None])
            self.loss = self.getCELossOp(self.logits, self.labels)
            self.graph.finalize()

    def buildNetwork(self):
        #   Construct an input layer:
        self.inputLayer = tf.placeholder(dtype=tf.float64, shape=[None, self.networkShape[0]])

        #   Build all hidden layers:
        prevLayer = self.inputLayer
        for i in range(1, len(self.networkShape)):
            with self.graph.name_scope("Layer{}".format(i)):
                newHiddenLayer = self.buildLayer(prevLayer, i)
                #   Add activation (can be non-differentiable! :D):
                newHiddenLayer = self.applyActivation(newHiddenLayer) if i != len(self.networkShape)-1 else tf.nn.sigmoid(newHiddenLayer)

                prevLayer = newHiddenLayer
        self.logits = prevLayer

        # Add softmax layer:
        yHat = tf.nn.softmax(prevLayer)
        return yHat

    def buildLayer(self, inputLayer, layerIndex):
        """
        Builds a layer in the network. Given a vector of input, creates W (weights) matrix and b (bias) vector, and
         creates input*W + b as output layer. Activation isn't added here so you can add whatever one you want later.
        :param inputLayer: vector of inputs. assuming rows are examples, and columns are features.
        :param layerIndex: integer representing the index of the layer (output). This will be used to determine the
        sizes.
        :return: the new layer Tensor (graph)
        """
        WShape = [self.networkShape[layerIndex - 1], self.networkShape[layerIndex]]
        W = tf.placeholder(name="weights", shape=WShape, dtype=tf.float64)
        bShape = [self.networkShape[layerIndex]]
        b = tf.placeholder(name="biases", shape=bShape, dtype=tf.float64)

        self.feedDict[W] = np.zeros(WShape)
        self.feedDict[b] = np.zeros(bShape)

        newLayer = tf.matmul(inputLayer, W) + b

        return newLayer

    @staticmethod
    def applyActivation(rawOutVec):
        #return tf.nn.relu(rawOutVec)
        return tf.nn.tanh(rawOutVec)
        # return NNFullyCon.stepActivation(rawOutVec)

    @staticmethod
    def stepActivation(rawOutVec):
        return tf.cast(rawOutVec > 0, dtype=tf.float64)

    def getParametersInfoList(self):
        """

        :return: list of tuples (shape, size) of all parameters (variables)
        """
        if self.parametersInfoList is not None:
            return self.parametersInfoList

        parametersInfoList = []
        with self.graph.as_default():
            for i in range(1, len(self.networkShape)):
                currentLayer = "Layer{}".format(i)
                W = self.graph.get_tensor_by_name(currentLayer + "/weights:0")
                b = self.graph.get_tensor_by_name(currentLayer + "/biases:0")
                parametersInfoList.append(self.sess.run([tf.shape(W), tf.size(W)]))
                parametersInfoList.append(self.sess.run([tf.shape(b), tf.size(b)]))
        self.parametersInfoList = parametersInfoList
        return parametersInfoList

    def updateParameters(self, parametersList):
        """
        Update the feed dict actually..
        :param parametersList: a list of all parameters in the same order as 'getParametersList' returns.
        :return:
        """
        with self.graph.as_default():
            for i in range(1, len(self.networkShape)):
                currentLayer = "Layer{}".format(i)

                self.feedDict[self.graph.get_tensor_by_name(currentLayer + "/weights:0")] = parametersList[2*(i-1)]
                self.feedDict[self.graph.get_tensor_by_name(currentLayer + "/biases:0")] = parametersList[2 * (i-1) + 1]

    def score(self, testExamples, testLabels):
        """
        Run the network on the given examples and return the accuracy based on the given labels.
        :param testExamples:
        :param testLabels:
        :return:
        """
        with self.graph.as_default():
            self.feedDict[self.inputLayer] = testExamples
            prediction = self.sess.run(self.prediction, self.feedDict)
            accuracy = np.sum((prediction == testLabels), dtype=float) / testLabels.size
        return accuracy

    @staticmethod
    def getPrediction(yHat):
        #   Assuming more than 1 example
        return tf.argmax(yHat, axis=1)

    def getCELossOp(self, logits, labels):
        ceLoss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        return ceLoss

    def getCELoss(self, testExamples, testLabels):
        with self.graph.as_default():
            self.feedDict[self.inputLayer] = testExamples
            self.feedDict[self.labels] = testLabels
            lossVal = self.sess.run(self.loss, self.feedDict)
        return lossVal*100

