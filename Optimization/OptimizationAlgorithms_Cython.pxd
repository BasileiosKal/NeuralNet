from OptBaseClass_Cy cimport Optimizer_cy
cimport numpy

cdef class GradientDescent(Optimizer_cy):
    cdef public str AlgorithmType
    cdef public mini_batch
    cdef public int iterations
    cdef public double learning_rate
    cdef void update_parameters_with_grad_descent(self, object Network)
    cdef float batch_epoch(self, object Network,double [:, :] X, long [:, :] Y)
    cpdef void Optimize(self, object Network,double [:, :] X, long [:, :] Y, print_result)


cpdef void gradient_descent_back_propagation(object Network,double [:, :] dZL,double [:, :] X)

cpdef float propagation(object Network,double [:, :] X, long [:, :] Y)
