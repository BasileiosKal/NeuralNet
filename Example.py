import data.Data_libraries.moons as data
from Optimization.OptimizationAlgorithms import GradientDescent, MomentumGradient
from Networks.NeuralNetworks import NeuralNetwork, FCLayerBuilder, InputLayer
from Utilities.Functions import RELU, Sigmoid
from Utilities.Gradient_Checking import gradient_checking
import numpy as np

relu = RELU()
sigmoid = Sigmoid()

X, Y = data.load_moon_data(n_samples=1000)

Hyper = {"iterations": 1000,
         "learning_rate": 0.01,
         "mini_batch": 64}

Input = {"type": InputLayer,
         "dim": X.shape[0]}

FClayer1 = {"Builder": FCLayerBuilder,
            "dim": 5,
            "activation": relu,
            "Regularization": None,
            "name": "FC Layer 1"}

FClayer2 = {"Builder": FCLayerBuilder,
            "dim": 4,
            "activation": relu,
            "Regularization": None,
            "name": "FC Layer 2"}

FClayer3 = {"Builder": FCLayerBuilder,
            "dim": 2,
            "activation": relu,
            "Regularization": None,
            "name": "FC Layer 3"}

FClayer4 = {"Builder": FCLayerBuilder,
            "dim": 1,
            "activation": sigmoid,
            "Regularization": None,
            "name": "FC Layer 4"}

layers = [Input, FClayer1, FClayer2, FClayer4]
Hyper_parameters = Hyper.values()

Network = NeuralNetwork(layers)
Optimizer = GradientDescent(*Hyper_parameters)
cost = Network.train(X, Y, Optimizer, plot_boundary=True, plot_cost=True)

# diff = gradient_checking(Network, X, Y, num_of_iterations=10000)
