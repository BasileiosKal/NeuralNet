import numpy as np


def compute_cost(A, Y):
    m = Y.shape[1]
    logprobs = np.multiply(-np.log(A), Y) + np.multiply(-np.log(1 - A), 1 - Y)
    costs = 1./m * np.sum(logprobs)
    return costs


# =============================================================================================== #
#                                   General optimizer class                                       #
# =============================================================================================== #
class Optimizer:
    def __init__(self, iterations, learning_rate, Mini_batch, Regularization):
        self.mini_batch = Mini_batch
        self.Regularization = Regularization
        self.iterations = iterations
        self.learning_rate = learning_rate

    def initialize(self, Network):
        pass

    @property
    def epoch(self):
        if self.mini_batch is None:
            if self.Regularization is None:
                Epoch = self.batch_epoch
            elif self.Regularization == "Dropout":
                Epoch = self.batch_epoch_dropout
        else:
            if self.Regularization is None:
                Epoch = self.mini_batch_epoch
            else:
                pass
        return Epoch

    def batch_epoch(self, Network, X, Y):
        pass

    def mini_batch_epoch(self, Network, X, Y):
        pass

    def batch_epoch_dropout(self, Network, X, Y):
        pass

