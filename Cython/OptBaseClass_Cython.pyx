import numpy as np
cimport numpy

cpdef double compute_cost(numpy.float64_t [:, :] A, numpy.int64_t [:, :] Y):
    cdef Py_ssize_t m = Y.shape[1]
    cdef double [:, :] Ones = np.ones((1, m))
    logprobs = np.multiply(-np.log(A), Y) + np.multiply(-np.log(np.subtract(Ones, A)), np.subtract(Ones, Y))
    return 1./m * np.sum(logprobs)


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


    def __init__(self, int iterations, double learning_rate, Mini_batch):
        self.mini_batch = Mini_batch
        self.iterations = iterations
        self.learning_rate = learning_rate


    def epoch(self):
        """Function to return the epoch to be used for optimization
        depending on the mini_batch hyper parameters of the optimization"""
        if self.mini_batch is None:
            Epoch = self.batch_epoch
        else:
            Epoch = self.mini_batch_epoch
        return Epoch
