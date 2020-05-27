import numpy as np


def compute_cost(A, Y):
    m = Y.shape[1]
    logprobs = np.multiply(-np.log(A), Y) + np.multiply(-np.log(1 - A), 1 - Y)
    costs = 1./m * np.sum(logprobs)
    return costs


# =============================================================================================== #
#                                      Base optimizer class                                       #
# =============================================================================================== #
class Optimizer:
    """Base class of the optimization algorithms. It contains the basic
    functions that all optimization algorithms have to over wright.
    These functions are accounting for every possible combination of
    mini batch or batch descent and regularization method.
    Inputs: iterations: [int32] Number of iterations
            learning_rate: [float32] The learning rate
            Mini_batch: [int] Mini batch size or [None] for batch gradient descent"""

    def __init__(self, iterations, learning_rate, Mini_batch):
        self.mini_batch = Mini_batch
        self.iterations = iterations
        self.learning_rate = learning_rate

    def initialize(self, Network):
        pass

    @property
    def epoch(self):
        """Function to return the epoch to be used for optimization
        depending on the mini_batch hyper parameters of the optimization"""
        if self.mini_batch is None:
            Epoch = self.batch_epoch
        else:
            Epoch = self.mini_batch_epoch
        return Epoch

    def batch_epoch(self, Network, X, Y):
        """Batch gradient descent with no regularization"""
        pass

    def mini_batch_epoch(self, Network, X, Y):
        """Mini batch gradient descent with no regularization"""
        pass
