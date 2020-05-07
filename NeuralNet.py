import numpy as np
from Utilities.Ploting import plot_decision_boundary
import matplotlib.pyplot as plt
import copy

# from Optimization.Algorithms import gradient_descent, Momentum
# from Optimization.Mini_Batch import *


class UntrainedLayer:
    def __init__(self, dims, function):
        self.dims = dims[1]
        self.prev_dims = dims[0]
        self.function = function
        self.W = None
        self.b = None
        self.linear_z = None
        self.activation = None

    def rand_initialize(self, constant=None):
        if constant:
            mul_parameter = constant   # mul_parameter: the parameter witch the weights will be multiplied
        else:
            mul_parameter = np.sqrt(2 / self.dims)
        self.W = np.random.randn(self.dims, self.prev_dims) * mul_parameter
        self.b = np.zeros((self.dims, 1))

    # Forward calculations
    def forward_calc(self, value):
        assert value.shape[0] == self.prev_dims
        assert self.W.shape == (self.dims, value.shape[0])
        assert self.b.shape == (self.dims, 1)

        if self.W.any():
            self.linear_z = np.dot(self.W, value) + self.b
            self.activation = self.function.calculate(self.linear_z)
        else:
            raise ValueError("parameters must be set")

        assert self.linear_z.shape == (self.dims, value.shape[1])
        assert self.activation.shape == self.linear_z.shape


class Layer(UntrainedLayer):
    def __init__(self, dims, function):
        super().__init__(dims, function)
        self.dW = None
        self.db = None
        self.dZ = None
        self.v_parameter = {}

    # Backward calculation
    def backward_calc(self, dZ, A_prev, Z_prev, prev_function, m):
        assert A_prev.shape == (self.prev_dims, m)
        self.dZ = dZ
        # self.dA = dA
        # self.dZ = self.dA * self.function.derivative(self.linear_z)
        self.dW = (1 / m) * np.dot(self.dZ, A_prev.T)
        self.db = (1 / m) * np.sum(self.dZ, axis=1, keepdims=True)
        prev_dZ = np.dot(self.W.T, self.dZ) * prev_function.derivative(Z_prev)

        assert self.dW.shape == self.W.shape
        assert self.db.shape == self.b.shape
        # assert self.dZ.shape == self.linear_z.shape
        # assert prev_dA.shape == (self.W.shape[1], self.dZ.shape[1])

        return prev_dZ


# ================================================================================================= #
#                                     Neural Network base class                                     #
# ================================================================================================= #
class NeuralNetwork:
    def __init__(self, layers_dim):
        self.layers_dim = layers_dim
        L = len(layers_dim)
        self.L = L


# ================================================================================================== #
#                                        Binary Classification                                       #
# ================================================================================================== #
class BinaryClassifier(NeuralNetwork):
    def __init__(self, layers_dim, functions_list):
        super().__init__(layers_dim)
        self.layers_functions = functions_list
        self.Layers = {}
        for layer in range(1, self.L):
            self.Layers[layer] = Layer([layers_dim[layer-1], layers_dim[layer]], functions_list[layer-1])
            self.Layers[layer].rand_initialize()

    @property
    def parameters(self):
        parameters = []
        for layer in range(1, self.L):    # keys from 1 to L-1 (not counting layer 0)
            parameters.append(self.Layers[layer].W)
            parameters.append(self.Layers[layer].b)
        return parameters

    @property
    def grads(self):
        grads = []
        for layer in range(1, self.L):  # keys from 1 to L (not counting layer 0)
            grads.append(self.Layers[layer].dW)
            grads.append(self.Layers[layer].db)
        return grads

    def front_propagation(self, X):
        A_prev = X
        for layer in range(1, self.L):
            current_layer = self.Layers[layer]
            current_layer.forward_calc(A_prev)
            A = current_layer.activation
            A_prev = A
        return A_prev

    def Train(self, X, Y, Optimization_algorithm, plot_boundary=False, plot_cost=False):
        """function for training the network
        Arguments:
            -X: np.array. The training data.
            -Y: mp.array. The true labels
            -Optimization_algorithm: The optimization algorithm used for training"""

        cost = Optimization_algorithm.Optimize(self, X, Y)
        self.plotting(X, Y, cost, plot_boundary, plot_cost, Optimization_algorithm.AlgorithmType)

        return cost

    def plotting(self, X, Y, cost, plot_boundary, plot_cost, title):
        if plot_boundary or plot_cost:
            fig = plt.figure()
        subplot_n_of_cols = plot_boundary + plot_cost

        if plot_boundary:
            plt.subplot(1, subplot_n_of_cols, 1)
            plt.title(title + ": Decision Boundary")
            axes = plt.gca()
            axes.set_xlim([-1.5, 2.5])
            axes.set_ylim([-1, 1.5])
            plot_decision_boundary(lambda x: self.predict(x.T), X, Y)

        if plot_cost:
            plt.subplot(1, subplot_n_of_cols, subplot_n_of_cols)
            plt.title(title + ": Cost")
            plt.plot(cost)
            plt.show()

    def predict(self, data):
        aL = self.front_propagation(data)
        predictions = (aL > 0.5)
        return predictions

    def reset(self, initial_parameters=None):
        for layer in self.Layers.values():
            layer.dZ = None
            layer.db = None
            layer.dW = None
            layer.v_parameter = {}

        if initial_parameters is not None:
            count = 0
            for layer in self.Layers.values():
                layer.W = copy.copy(initial_parameters[count])
                layer.b = copy.copy(initial_parameters[count+1])
                count += 2
        else:
            for layer in self.Layers.values():
                layer.rand_initialize()
