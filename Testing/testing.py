from NeuralNet import BinaryClassifier
from NN_utils import gradient_checking
import matplotlib.pyplot as plt
import Functions
import Testing.TestCases as TestCases
import numpy as np
from Optimization import gradient_descent, momentum_gradient_descent
from data.Data_libraries import moons

# Test cases
X2 = TestCases.X2
Y2 = TestCases.Y2
[W1, b1, W2, b2, W3, b3] = TestCases.parameters
[dW3, db3, dW2, db2, dW1, db1] = TestCases.grads
[dZ3, dZ2, dZ1] = TestCases.dZ
[dA2, dA1] = TestCases.dA

# Functions for the activation functions of the layers
re = Functions.RELU(np.array([[0]]))
si = Functions.Sigmoid(0)
functions = [re for i in range(2)]
functions.append(si)

# The network for the testing
TestingNetwork = BinaryClassifier([X2.shape[0], 5, 3, 1], functions)
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


def test_train():
    epsilon = 1e-7
    gradient_descent([TestingNetwork], X2, Y2, 1)
    assert (abs(a1.dW - dW1) < epsilon).all()
    assert (abs(a2.dW - dW2) < epsilon).all()
    assert (abs(a3.dW - dW3) < epsilon).all()

    assert (abs(a1.db - db1) < epsilon).all()
    assert (abs(a2.db - db2) < epsilon).all()
    assert (abs(a3.db - db3) < epsilon).all()

    assert (abs(a1.dZ - dZ1) < epsilon).all()
    assert (abs(a2.dZ - dZ2) < epsilon).all()
    assert (abs(a3.dZ - dZ3) < epsilon).all()


