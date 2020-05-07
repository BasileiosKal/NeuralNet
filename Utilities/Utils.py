import numpy as np


def shuffle_data(X, Y):
    m = X.shape[1]  # number of training examples

    # Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))
    return shuffled_X, shuffled_Y


def Parallel_Training(X, Y, iterations, Network_list, Optimizer_list, plot_boundary=False, plot_cost=False):
    num_of_nets = len(Network_list)
    cost = np.ones((num_of_nets, iterations))

    # Initialize the optimizers
    for i in range(num_of_nets):
        optimizer = Optimizer_list[i]
        network = Network_list[i]
        optimizer.initialize(network)

    # Train the networks
    for iteration in range(iterations):
        for ii in range(num_of_nets):
            epoch = Optimizer_list[ii].epoch
            cost_value = epoch(Network_list[ii], X, Y)
            cost[ii, iteration] = cost_value

    # Plotting the decision boundary and/or cost and printing the results
    for i in range(num_of_nets):
        net = Network_list[i]
        opt = Optimizer_list[i]
        net.plotting(X, Y, cost[i], plot_boundary, plot_cost, opt.AlgorithmType)
        print("--------------------------------------------------------------------------------")
        print(opt.AlgorithmType + ": cost after " + str(iterations) + " iterations: " + str(cost[i][-1]))
        print("Accuracy with " + opt.AlgorithmType + ": " + str(1 - ((np.sum(abs(net.predict(X) - Y))) / Y.shape[1])))
        print("--------------------------------------------------------------------------------")
    return cost

