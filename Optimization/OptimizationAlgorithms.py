import numpy as np
from Optimization.OptBaseClass import Optimizer, compute_cost
from Optimization.Mini_Batch import create_mini_batches


# =============================================================================================== #
#                                      gradient descent                                           #
# =============================================================================================== #
def gradient_descent_back_propagation(Network, dZL, X):
    """current_layer: the layer that back_propagation passes through.
                      Specifically the layer in which we calculate the grads.
           prev_layer: The previous layer in the back propagation direction.
                       Specifically the layer that we have calculate the grads.
           Example: if the current is the 3th hidden layer of the network
                    then prev_layer will be the 2nd layer. """

    m = dZL.shape[1]
    # Setting up the input layer "activation" as X
    inputLayer = Network.Layers[0]
    inputLayer.activation = X

    # Setting up the current layer that back propagation passes
    # to be initially the last layer of the network with dZ = dZL
    current_layer = Network.Layers[-1]
    current_layer.dZ = dZL

    for layer in reversed(Network.Layers[1:-1]):
        prev_layer = layer

        current_layer.backward_calc(prev_layer, m)
        prev_layer.dZ = np.dot(current_layer.W.T, current_layer.dZ) * prev_layer.function.derivative(prev_layer.linear_z)

        current_layer = prev_layer

    # For the last layer.
    Network.Layers[1].backward_calc(inputLayer, m)


def gradient_descent_epoch(Network, X, Y):
    # front propagation
    Y_hat = Network.front_propagation(X)
    # cost
    cost_value = compute_cost(Y_hat, Y)
    # back propagation
    dZL = Y_hat - Y
    gradient_descent_back_propagation(Network, dZL, X)
    return cost_value


class GradientDescent(Optimizer):
    """Gradient descent optimizer
    inputs: iterations: [int] number of iterations
            learning_rate: [float]
            Mini_batch: [int] the size of the mini batches or [None] for batch descent
    example:
        Hyper = {"iterations": 1000,
                 "learning_rate": 0.01,
                 "mini_batch": 64}
        Hyper_parameters = Hyper.values()

        Optimizer = GradientDescent(*Hyper_parameters)
        cost = Network.Train(X, Y, Optimizer)"""

    def __init__(self, iterations, learning_rate, Mini_batch):
        super().__init__(iterations, learning_rate, Mini_batch)
        self.AlgorithmType = "Gradient descent"

    def update_parameters_with_grad_descent(self, Network):
        for ii in range(Network.L - 1):
            Network.Layers[ii + 1].W -= self.learning_rate * Network.Layers[ii + 1].dW
            Network.Layers[ii + 1].b -= self.learning_rate * Network.Layers[ii + 1].db

    def batch_epoch(self, Network, X, Y):
        cost_val = gradient_descent_epoch(Network, X, Y)
        self.update_parameters_with_grad_descent(Network)
        return cost_val

    def mini_batch_epoch(self, Network, X, Y):
        mini_batches = create_mini_batches(X, Y, self.mini_batch)
        cost_total = 0
        m = X.shape[1]
        for batch in mini_batches:
            (batch_X, batch_Y) = batch
            # one epoch using momentum on the mini batch.
            batch_cost = gradient_descent_epoch(Network, batch_X, batch_Y)
            cost_total += batch_cost
            # update the parameters.
            self.update_parameters_with_grad_descent(Network)
        cost_avg = cost_total / m
        return cost_avg

    def Optimize(self, Network, X, Y, print_result):
        cost = []
        self.initialize(Network)
        Epoch = self.epoch
        for iteration in range(self.iterations):
            cost_value = Epoch(Network, X, Y)
            cost.append(cost_value)

        if print_result:
            print("--------------------------------------------------------------------------------")
            print("Momentum: cost after " + str(self.iterations) + " iterations: " + str(cost[-1]))
            print("Accuracy with Momentum: " + str(1 - ((np.sum(abs(Network.predict(X) - Y))) / Y.shape[1])))
            print("--------------------------------------------------------------------------------")
        return cost


# =============================================================================================== #
#                              gradient descent with momentum                                     #
# =============================================================================================== #
def initialize_momentum(Network):
    """initializing the momentum parameters"""
    for layer in Network.Layers[1:]:
        layer.v_parameter["dW"] = np.zeros(layer.W.shape)
        layer.v_parameter["db"] = np.zeros(layer.b.shape)


class MomentumGradient(Optimizer):
    """Momentum gradient descent optimizer
        inputs: iterations: [int] number of iterations
                learning_rate: [float]
                Mini_batch: [int] the size of the mini batches or [None] for batch descent
        example:
            Hyper = {"iterations": 1000,
                     "learning_rate": 0.01,
                     "mini_batch": 64}
            Hyper_parameters = Hyper.values()

            Optimizer = MomentumGradient(*Hyper_parameters)
            cost = Network.Train(X, Y, Optimizer)"""

    def __init__(self, beta, iterations, learning_rate, Mini_batch=None):
        super().__init__(iterations, learning_rate, Mini_batch)
        self.beta = beta
        self.AlgorithmType = "Gradient descent with Momentum"

    def initialize(self, Network):
        initialize_momentum(Network)

    def backward_calculations(self, layer, prev_layer, m):
        """backward calculations for when the back propagation
        passes from the specified layer to the prev_layer.

        It uses the layers method for calculating the gradients
        (backward_calc) of the layer adding the calculation of
        the momentum parameter afterwards"""

        assert prev_layer.activation.shape == (layer.dimensions["in"], m)

        layer.backward_calc(prev_layer, m)

        layer.v_parameter["dW"] = (self.beta * layer.v_parameter["dW"]) + (1 - self.beta) * layer.dW
        layer.v_parameter["db"] = (self.beta * layer.v_parameter["db"]) + (1 - self.beta) * layer.db

        assert layer.dW.shape == layer.W.shape
        assert layer.db.shape == layer.b.shape

    def back_propagation(self, Network, dZL, X):
        """current_layer: the layer that back_propagation passes through.
                          Specifically the layer in which we calculate the grads.
           prev_layer: The previous layer in the back propagation direction.
                       Specifically the layer that we have calculate the grads.
           Example: if the current is the 3th hidden layer of the network
                    then prev_layer will be the 2nd layer. """

        m = dZL.shape[1]
        # Setting up the input layer "activation" as X
        inputLayer = Network.Layers[0]
        inputLayer.activation = X

        # Setting up the current layer that back propagation passes
        # to be initially the last layer of the network with dZ = dZL
        current_layer = Network.Layers[-1]
        current_layer.dZ = dZL

        for layer in reversed(Network.Layers[1:-1]):
            prev_layer = layer

            self.backward_calculations(current_layer, prev_layer, m)
            prev_layer.dZ = np.dot(current_layer.W.T, current_layer.dZ) * prev_layer.function.derivative(
                prev_layer.linear_z)

            current_layer = prev_layer

        # For the last layer.
        self.backward_calculations(Network.Layers[1], inputLayer, m)

    def propagation(self, Network, X, Y):
        # front propagation
        Y_hat = Network.front_propagation(X)
        # cost
        cost_value = compute_cost(Y_hat, Y)
        # back propagation
        dZL = Y_hat - Y
        self.back_propagation(Network, dZL, X)
        return cost_value

    def update_parameters(self, Network):
        for layer in Network.Layers[1:]:
            layer.W -= self.learning_rate * layer.v_parameter["dW"]
            layer.b -= self.learning_rate * layer.v_parameter["db"]

    def batch_epoch(self, Network, X, Y):
        calc_cost = self.propagation(Network, X, Y)
        # update parameters
        self.update_parameters(Network)
        return calc_cost

    def mini_batch_epoch(self, Network, X, Y):
        m = X.shape[1]
        mini_batches = create_mini_batches(X, Y, self.mini_batch)
        cost_total = 0
        for batch in mini_batches:
            (batch_X, batch_Y) = batch
            # one epoch using momentum on the mini batch.
            batch_cost = self.propagation(Network, batch_X, batch_Y)
            cost_total += batch_cost
            # update the parameters.
            self.update_parameters(Network)
        cost_avg = cost_total / m
        return cost_avg

    def Optimize(self, Network, X, Y, print_result):
        cost = []
        self.initialize(Network)
        Epoch = self.epoch
        for iteration in range(self.iterations):
            cost_value = Epoch(Network, X, Y)
            cost.append(cost_value)
        if print_result:
            print("--------------------------------------------------------------------------------")
            print("Momentum: cost after " + str(self.iterations) + " iterations: " + str(cost[-1]))
            print("Accuracy with Momentum: " + str(1 - ((np.sum(abs(Network.predict(X) - Y))) / Y.shape[1])))
            print("--------------------------------------------------------------------------------")
        return cost
