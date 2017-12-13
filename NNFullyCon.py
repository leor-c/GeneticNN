import GeneticNN
import tensorflow as tf
import numpy as np

class NNFullyCon(GeneticNN.NNGeneticModel):
    """
    A Class for fully connected NN that obeys the GeneticNN requirements.
    """

    def __init__(self, networkShape):
        """
        Constructs a fully connected NN with given shape. The shape includes input & out dimensions with at least 1
        hidden layer.
        :param networkShape: a tuple/list/nparray with at least 3 dimensions.
        """
        self.networkShape = networkShape
        self.inputLayer = None
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.yHat = self.buildNetwork()
            self.prediction = self.getPrediction(self.yHat)
            #   Init variables:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())


    def buildNetwork(self):
        #   Construct an input layer:
        self.inputLayer = tf.placeholder(dtype=tf.float64, shape=[None, self.networkShape[0]])

        #   Build all hidden layers:
        prevLayer = self.inputLayer
        for i in range(1, len(self.networkShape)):
            with tf.variable_scope("Layer{}".format(i), reuse=tf.AUTO_REUSE):
                newHiddenLayer = self.buildLayer(prevLayer, i)
                #   Add activation (can be non-differentiable! :D):
                newHiddenLayer = self.applyActivation(newHiddenLayer)

                prevLayer = newHiddenLayer

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
        W = tf.get_variable("weights", [self.networkShape[layerIndex - 1], self.networkShape[layerIndex]], initializer=tf.random_normal_initializer(), dtype=tf.float64)
        b = tf.get_variable("biases", [self.networkShape[layerIndex]], initializer=tf.zeros_initializer(), dtype=tf.float64)

        newLayer = tf.matmul(inputLayer, W) + b

        return newLayer

    @staticmethod
    def applyActivation(rawOutVec):
        return tf.nn.relu(rawOutVec)
        #return NNFullyCon.stepActivation(rawOutVec)

    @staticmethod
    def stepActivation(rawOutVec):
        return tf.cast(rawOutVec > 0, dtype=tf.float64)


    def getParametersList(self):
        parametersList = []
        with self.graph.as_default():
            for i in range(1, len(self.networkShape)):
                with tf.variable_scope("Layer{}".format(i), reuse=True):
                    W = tf.get_variable("weights", [self.networkShape[i - 1], self.networkShape[i]], dtype=tf.float64)
                    b = tf.get_variable("biases", [self.networkShape[i]], dtype=tf.float64)
                    parametersList.append(self.sess.run(W))
                    parametersList.append(self.sess.run(b))
        return parametersList

    def updateParameters(self, parametersList):
        """
        Update the parameters/weights of the network using the given parameters list.
        :param parametersList: a list of all parameters in the same order as 'getParametersList' returns.
        :return:
        """
        with self.graph.as_default():
            for i in range(1, len(self.networkShape)):
                with tf.variable_scope("Layer{}".format(i), reuse=True):
                    W = tf.get_variable("weights", [self.networkShape[i - 1], self.networkShape[i]], dtype=tf.float64)
                    b = tf.get_variable("biases", [self.networkShape[i]], dtype=tf.float64)

                    self.sess.run(tf.assign(W, parametersList[2*(i-1)]))
                    self.sess.run(tf.assign(b, parametersList[2 * (i-1) + 1]))

    def score(self, testExamples, testLabels):
        """
        Run the network on the given examples and return the accuracy based on the given labels.
        :param testExamples:
        :param testLabels:
        :return:
        """
        with self.graph.as_default():
            prediction = self.sess.run(self.prediction, {self.inputLayer: testExamples})
            accuracy = np.sum((prediction == testLabels), dtype=float) / testLabels.size
        return accuracy

    @staticmethod
    def getPrediction(yHat):
        #   Assuming more than 1 example
        return tf.argmax(yHat, axis=1)




