import matplotlib.pyplot as plt
import numpy as np


def plot_decision_boundary(model, X, y):
    """Plotting the decision boundary when the
    data X are 2D.
    Inputs:
    -model: The Neural network model for mapping data X to
            the networks output. Usually will have te form:
            lambda x: self.predict(x.T)
    -X: Data
    -Y: True labels of the data
    """
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y.reshape((y.shape[1], )), cmap=plt.cm.Spectral)
