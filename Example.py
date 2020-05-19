import data.Data_libraries.moons as data
from Optimization.OptimizationAlgorithms import GradientDescent, MomentumGradient
from NeuralNet import NeuralNetwork
from Utilities.Functions import RELU,Sigmoid
import numpy as np

relu = RELU(np.array([[0]]))
sigmoid = Sigmoid(0)

X,Y = data.load_moon_data(n_samples=1000)

Hyper = {"iterations": 1000,
         "learning_rate": 0.01,
         "mini_batch": 64}

Input = {"dim": X.shape[0]}

FClayer1 = {"type": 'FC',
            "dim": 5,
            "activation": relu,
            "Regularization": None}
FClayer2 = {"type": 'FC',
            "dim": 2,
            "activation": relu,
            "Regularization": None}
FClayer3 = {"type": 'FC',
            "dim": 1,
            "activation": sigmoid,
            "Regularization": None}
layers = [Input, FClayer1, FClayer2, FClayer3]
Hyper_parameters = Hyper.values()

Network = NeuralNetwork(layers)
Optimizer = MomentumGradient(0.9, *Hyper_parameters)
cost = Network.Train(X, Y, Optimizer, plot_boundary=True, plot_cost=True)
