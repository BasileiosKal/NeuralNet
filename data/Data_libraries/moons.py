import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets


def load_moon_data(n_samples=300, sample_noise=0.2):
    train_X, train_Y = sklearn.datasets.make_moons(n_samples, noise=sample_noise)
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    return train_X, train_Y


def plot_data(train_X, train_Y):
    plt.scatter(train_X[0, :], train_X[1, :], c=train_Y.reshape((train_Y.shape[1], )), s=40, cmap=plt.cm.Spectral)
