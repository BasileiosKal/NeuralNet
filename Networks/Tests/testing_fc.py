from Networks.NeuralNetworks import NeuralNetwork, FCLayerBuilder
from Utilities import Functions
import Testing.TestCases as TestCases
import numpy as np
from Optimization.OptimizationAlgorithms import GradientDescent
import unittest

X2 = TestCases.X2
Y2 = TestCases.Y2

# Functions for the activation functions of the layers
relu = Functions.RELU()
sigmoid = Functions.Sigmoid()

# The network for the testing
Input = {"type": "Input", "dim": X2.shape[0]}

FClayer1 = {"Builder": FCLayerBuilder,
            "dim": 5,
            "activation": relu,
            "Regularization": None,
            "name": "FC layer 1"}

FClayer2 = {"Builder": FCLayerBuilder,
            "dim": 3,
            "activation": relu,
            "Regularization": None,
            "name": "FC layer 2"}

FClayer3 = {"Builder": FCLayerBuilder,
            "dim": 1,
            "activation": sigmoid,
            "Regularization": None,
            "name": "FC layer 3"}

layers = [Input, FClayer1, FClayer2, FClayer3]
TestingNetwork = NeuralNetwork(layers)
a1 = TestingNetwork.Layers[1]
a2 = TestingNetwork.Layers[2]
a3 = TestingNetwork.Layers[3]

Opt = GradientDescent(1, 0.00075, Mini_batch=None)
TestingNetwork.train(X2, Y2, Opt, print_result=False)


class TestForwardCalc(unittest.TestCase):
    """Test for the forward  calculations of a FC layer.
    Using the TestingNetwork defined above with 2 hidden FC layers of
    dimensions 5, 3 and input of shape (2, m)
    ===========
    NOTE: a1, a2, a3 denote the layers of the network.

    """
    def test_W_dimensions(self):
        """Testing the parameter W dimensions"""
        self.assertEqual(X2.shape[0], a1.dimensions["in"])
        self.assertEqual(a1.W.shape, (5, X2.shape[0]))
        self.assertEqual(a2.W.shape, (3, a1.dimensions["out"]))
        self.assertEqual(a3.W.shape, (1, a2.dimensions["out"]))

    def test_b_dimensions(self):
        """Testing the parameter b dimensions"""
        self.assertEqual(a1.b.shape, (5, 1))
        self.assertEqual(a2.b.shape, (3, 1))
        self.assertEqual(a3.b.shape, (1, 1))

    def test_linear_z_dimensions(self):
        """Testing the linear activation z dimensions"""
        self.assertEqual(a1.linear_z.shape, (5, X2.shape[1]))
        self.assertEqual(a2.linear_z.shape, (3, X2.shape[1]))
        self.assertEqual(a3.linear_z.shape, (1, X2.shape[1]))

    def test_activation_dimensions(self):
        """Testing the activation dimensions"""
        self.assertEqual(a1.activation.shape, (5, X2.shape[1]))
        self.assertEqual(a2.activation.shape, (3, X2.shape[1]))
        self.assertEqual(a3.activation.shape, (1, X2.shape[1]))


class TestBackwardCalc(unittest.TestCase):
    """Test for the forward  calculations of a FC layer.
    Using the TestingNetwork defined above with 2 hidden FC layers of
    dimensions 5, 3 and input of shape (2, m)
    ===========
    NOTE: a1, a2, a3 denote the layers of the network.

    """
    def test_grads_dimensions(self):
        """Testing the dimensions of the grads"""
        m = X2.shape[1]
        self.assertEqual(a1.dW.shape, (5, X2.shape[0]))
        self.assertEqual(a2.dW.shape, (3, 5))
        self.assertEqual(a3.dW.shape, (1, 3))

        self.assertEqual(a1.db.shape, (5, 1))
        self.assertEqual(a2.db.shape, (3, 1))
        self.assertEqual(a3.db.shape, (1, 1))

        self.assertEqual(a1.dZ.shape, (5, m))
        self.assertEqual(a2.dZ.shape, (3, m))
        self.assertEqual(a3.dZ.shape, (1, m))

        self.assertEqual(a1.activation.shape, (5, m))
        self.assertEqual(a2.activation.shape, (3, m))
        self.assertEqual(a3.activation.shape, (1, m))
