from NeuralNet.Utilities.Utils import shuffle_data
import math


def create_mini_batches(X, Y, mini_match_size):
    shuffled_X, shuffled_Y = shuffle_data(X, Y)
    mini_batches = []
    m = X.shape[1]
    # The number of mini batches of size = mini_match_size. Not counting the last one
    num_complete_batches = math.floor(m/mini_match_size)
    # Creating the complete mini batches
    for i in range(num_complete_batches):
        mini_batch_X = shuffled_X[:, i*mini_match_size:(i+1)*mini_match_size]
        mini_batch_Y = shuffled_Y[:, i*mini_match_size:(i+1)*mini_match_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % mini_match_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_batches * mini_match_size:m]
        mini_batch_Y = shuffled_Y[:, num_complete_batches * mini_match_size:m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def mini_batch_optimize(Network, epoch, update_parameters, X, Y, batch_size, m):
    mini_batches = create_mini_batches(X, Y, batch_size)
    cost_total = 0
    for batch in mini_batches:
        (batch_X, batch_Y) = batch
        # one epoch using momentum on the mini batch.
        batch_cost = epoch(Network, batch_X, batch_Y)
        cost_total += batch_cost
        # update the parameters.
        update_parameters(Network)
    cost_avg = cost_total / m
    return cost_avg
