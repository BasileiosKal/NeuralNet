from NeuralNet import NeuralNetwork
from Utilities import Functions
import Testing.TestCases as TestCases
import numpy as np
from Optimization.OptimizationAlgorithms import GradientDescent
import unittest

# Test cases
X2 = TestCases.X2
Y2 = TestCases.Y2
[W1, b1, W2, b2, W3, b3] = TestCases.parameters
[dW3, db3, dW2, db2, dW1, db1] = TestCases.grads
[dZ3, dZ2, dZ1] = TestCases.dZ
[dA2, dA1] = TestCases.dA

# Functions for the activation functions of the layers
relu = Functions.RELU(np.array([[0]]))
sigmoid = Functions.Sigmoid(0)
functions = [relu for i in range(2)]
functions.append(sigmoid)

# The network for the testing
Input = {"dim": X2.shape[0]}

FClayer1 = {"type": 'FC',
            "dim": 5,
            "activation": relu,
            "Regularization": None}
FClayer2 = {"type": 'FC',
            "dim": 3,
            "activation": relu,
            "Regularization": None}
FClayer3 = {"type": 'FC',
            "dim": 1,
            "activation": sigmoid,
            "Regularization": None}

layers = [Input, FClayer1, FClayer2, FClayer3]
TestingNetwork = NeuralNetwork(layers)
a1 = TestingNetwork.Layers[1]
a2 = TestingNetwork.Layers[2]
a3 = TestingNetwork.Layers[3]

# Setting the initial parameters
a1.W = W1
a1.b = b1
a2.W = W2
a2.b = b2
a3.W = W3
a3.b = b3

# gradient_checking(TestingNetwork, X2, Y2, 100)
# print("Cost after training: ", TestingNetwork.cost[-1])
# plt.plot(TestingNetwork.cost)
# plt.show()


class TestOptimizer(unittest.TestCase):
    def test_GradientDescent(self):
        epsilon = 1e-7
        Opt = GradientDescent(1, 0.00075, Mini_batch=None)
        TestingNetwork.Train(X2, Y2, Opt, print_result=False)
        assert (abs(a1.dW - dW1) < epsilon).all()
        assert (abs(a2.dW - dW2) < epsilon).all()
        assert (abs(a3.dW - dW3) < epsilon).all()

        assert (abs(a1.db - db1) < epsilon).all()
        assert (abs(a2.db - db2) < epsilon).all()
        assert (abs(a3.db - db3) < epsilon).all()

        assert (abs(a1.dZ - dZ1) < epsilon).all()
        assert (abs(a2.dZ - dZ2) < epsilon).all()
        assert (abs(a3.dZ - dZ3) < epsilon).all()


