cimport numpy

cpdef double compute_cost(numpy.float64_t [:, :] A, numpy.int64_t [:, :] Y)
cdef class Optimizer_cy:
    cpdef epoch(self)
