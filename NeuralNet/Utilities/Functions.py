import numpy as np


class FunctionsBase:
    def __init__(self):
        self.value = None
        self.deriv = None


class Sigmoid(FunctionsBase):
    def __init__(self):
        super().__init__()

    def calculate(self, value):
        self.value = 1/(1+np.exp(-value))
        return self.value

    def derivative(self, x):
        self.deriv = self.calculate(x) * (1-self.calculate(x))
        return self.deriv


class Identity(FunctionsBase):
    def __init__(self):
        super().__init__()

    def calculate(self, value):
        self.value = value
        return self.value

    def derivative(self, x):
        self.deriv = 1 + 0*x
        return self.deriv


class RELU(FunctionsBase):
    def __init__(self):
        super().__init__()

    def calculate(self, value):
        D = (value > 0).astype(int)
        self.value = value*D
        return self.value

    def derivative(self, x):
        self.deriv = (x > 0).astype(int)
        return self.deriv
