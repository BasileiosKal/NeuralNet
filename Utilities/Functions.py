import numpy as np


class Sigmoid:
    def __init__(self, value):
        self.value = self.calculate(value)
        self.deriv = self.derivative(value)

    def calculate(self, value):
        self.value = 1/(1+np.exp(-value))
        return self.value

    def derivative(self, x):
        self.deriv = self.calculate(x) * (1-self.calculate(x))
        return self.deriv


class Identity:
    def __init__(self, value):
        self.value = self.calculate(value)
        self.deriv = self.derivative(value)

    def calculate(self, value):
        self.value = value
        return self.value

    def derivative(self, x):
        self.deriv = 1 + 0*x
        return self.deriv


class RELU:
    def __init__(self, value):
        self.value = self.calculate(value)
        self.deriv = self.derivative(value)

    def calculate(self, value):
        D = (value > 0).astype(int)
        self.value = value*D
        return self.value

    def derivative(self, x):
        self.deriv = (x > 0).astype(int)
        return self.deriv
