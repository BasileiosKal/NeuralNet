import numpy as np
from Optimization.Optimizers import Optimizer, compute_cost
from Optimization.Mini_Batch import create_mini_batches, mini_batch_optimize


# =============================================================================================== #
#                                      gradient descent                                           #
# =============================================================================================== #
def gradient_descent_back_propagation(Network, dZ, X):
    m = dZ.shape[1]
    for layer in reversed(range(2, Network.L)):
        prev_layer = Network.Layers[layer-1]
        current_layer = Network.Layers[layer]

        A_prev = prev_layer.activation
        Z_prev = prev_layer.linear_z
        prev_function = prev_layer.function
        dZ = current_layer.backward_calc(dZ, A_prev, Z_prev, prev_function, m)
    # For the last layer.
    current_layer = Network.Layers[1]
    current_layer.dZ = dZ
    current_layer.dW = (1/m)*np.dot(dZ, X.T)
    current_layer.db = (1/m)*np.sum(dZ, axis=1, keepdims=True)


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
    def __init__(self, iterations, learning_rate, Mini_batch, Regularization=None):
        super().__init__(iterations, learning_rate, Mini_batch, Regularization)
        self.AlgorithmType = "Gradient descent"

    def update_parameters_with_grad_descent(self, Network):
        for ii in range(Network.L - 1):
            Network.Layers[ii + 1].W -= self.learning_rate * Network.Layers[ii + 1].dW
            Network.Layers[ii + 1].b -= self.learning_rate * Network.Layers[ii + 1].db

    def batch_gradient_descent(self, Network, X, Y):
        cost = []
        for i in range(self.iterations):
            cost_val = gradient_descent_epoch(Network, X, Y)
            cost.append(cost_val)
            self.update_parameters_with_grad_descent(Network)

        print("Gradient Descent: Cost after " + str(self.iterations) + " iterations: " + str(cost[-1]))
        print("Accuracy with Gradient Descent: " + str(1 - ((np.sum(abs(Network.predict(X) - Y))) / Y.shape[1])))
        print("--------------------------------------------------------------------------------")
        return cost

    def mini_batch_gradient_descent(self, Network, X, Y):
        m = X.shape[1]
        batch_size = self.mini_batch
        costs = []
        epoch = gradient_descent_epoch
        update_parameters = self.update_parameters_with_grad_descent
        for iteration in range(self.iterations):
            cost_avg = mini_batch_optimize(Network, epoch, update_parameters, X, Y, batch_size, m)
            costs.append(cost_avg)

        print("Gradient Descent: cost after " + str(self.iterations) + " iterations: " + str(costs[-1]))
        print("Accuracy with Gradient Descent: " + str(1 - ((np.sum(abs(Network.predict(X) - Y))) / Y.shape[1])))
        print("--------------------------------------------------------------------------------")
        return costs

    def Optimize(self, Network, X, Y):
        cost = None
        if self.mini_batch is False:
            if self.Regularization is None:
                cost = self.batch_gradient_descent(Network, X, Y)
            else:
                pass
        else:
            if self.Regularization is None:
                cost = self.mini_batch_gradient_descent(Network, X, Y)
            else:
                pass
        return cost


# =============================================================================================== #
#                              gradient descent with momentum                                     #
# =============================================================================================== #
def initialize_momentum(Network):
    for layer in Network.Layers.values():
        layer.v_parameter["dW"] = np.zeros(layer.W.shape)
        layer.v_parameter["db"] = np.zeros(layer.b.shape)


class MomentumGradient(Optimizer):
    def __init__(self, beta, iterations, learning_rate, Mini_batch=None, Regularization=None):
        super().__init__(iterations, learning_rate, Mini_batch, Regularization)
        self.beta = beta
        self.AlgorithmType = "Gradient descent with Momentum"

    def initialize(self, Network):
        initialize_momentum(Network)

    def backward_calculations(self, layer, dZ, A_prev, Z_prev, prev_function, m):
        assert A_prev.shape == (layer.prev_dims, m)
        prev_dZ = layer.backward_calc(dZ, A_prev, Z_prev, prev_function, m)

        layer.v_parameter["dW"] = (self.beta * layer.v_parameter["dW"]) + (1 - self.beta) * layer.dW
        layer.v_parameter["db"] = (self.beta * layer.v_parameter["db"]) + (1 - self.beta) * layer.db

        assert layer.dW.shape == layer.W.shape
        assert layer.db.shape == layer.b.shape

        return prev_dZ

    def back_propagation(self, Network, dZ, X):
        m = dZ.shape[1]
        if self.Regularization is None:
            backward_calculation = self.backward_calculations
        else:
            pass
        for layer in reversed(range(2, Network.L)):
            prev_layer = Network.Layers[layer - 1]
            current_layer = Network.Layers[layer]

            A_prev = prev_layer.activation
            Z_prev = prev_layer.linear_z
            prev_function = prev_layer.function
            dZ = backward_calculation(current_layer, dZ, A_prev, Z_prev, prev_function, m)
            #dZ = current_layer.backward_calc(dZ, A_prev, Z_prev, prev_function, m)

        # For the last layer.
        current_layer = Network.Layers[1]
        current_layer.dZ = dZ
        current_layer.dW = (1 / m) * np.dot(dZ, X.T)
        current_layer.db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        current_layer.v_parameter["dW"] = (self.beta*current_layer.v_parameter["dW"]) + (1 - self.beta)*current_layer.dW
        current_layer.v_parameter["db"] = (self.beta*current_layer.v_parameter["db"]) + (1 - self.beta)*current_layer.db

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
        for ii in range(Network.L - 1):
            Network.Layers[ii + 1].W -= self.learning_rate * Network.Layers[ii + 1].v_parameter["dW"]
            Network.Layers[ii + 1].b -= self.learning_rate * Network.Layers[ii + 1].v_parameter["db"]

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

    def Optimize(self, Network, X, Y):
        cost = []
        self.initialize(Network)
        Epoch = self.epoch
        for iteration in range(self.iterations):
            cost_value = Epoch(Network, X, Y)
            cost.append(cost_value)

        print("--------------------------------------------------------------------------------")
        print("Momentum: cost after " + str(self.iterations) + " iterations: " + str(cost[-1]))
        print("Accuracy with Momentum: " + str(1 - ((np.sum(abs(Network.predict(X) - Y))) / Y.shape[1])))
        print("--------------------------------------------------------------------------------")
        return cost
