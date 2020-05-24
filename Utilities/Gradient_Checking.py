import numpy as np
from Optimization.OptimizationAlgorithms import GradientDescent


def parameters_to_vectors(parameters):
    """First helper function for gradient checking.
    Unroll the parameters of the NN to a single vector.
    Made for grads testing
    """
    num_of_parameters = len(parameters)
    vector = None   # this doesnt have to be set to None
    for ii in range(num_of_parameters):
        par_vector = parameters[ii].reshape((-1, 1))
        if ii == 0:
            vector = par_vector
        else:
            vector = np.concatenate((vector, par_vector), axis=0)
    return vector


def vector_to_parameters(vector, dims):
    """Second helper function for gradient checking.
    The inverse of the previous function. Returning
    the parameters with given dimensions from a vector
    arguments:
    -vector: a numpy array of shape (n,1)
    -dims: a list with the dimensions of the parameters. Containing 1x2 arrays with the
           shape of each parameter, for example: dims = [[3,4], [3,1]] for two parameters, a weights 3x4 matrix and a
           bias parameter b of shape 4x1
           """
    n = len(dims)
    parameters = []
    end_dims = 0
    for ii in range(n):
        # if the parameter is b or db, with shape [n, 1]
        if dims[ii][1] == 1:
            parameter = vector[end_dims:end_dims+dims[ii][0]].reshape((dims[ii][0], dims[ii][1]))
            end_dims += dims[ii][0]
        else:   # if the parameter is W or dW
            parameter = vector[end_dims:end_dims+dims[ii][0]*dims[ii][1]].reshape((dims[ii][0], dims[ii][1]))
            end_dims += dims[ii][0]*dims[ii][1]

        parameters.append(parameter)
    return parameters


def Forward_calculations(X, parameters, functions):
    """Third helper function for gradient checking.
    Calculate forward propagation, given the parameters
    and functions of the network.
    Inputs:
    -parameters: [list] in the form of [W1, b1, ...]
                 where W1, b1, ... are np.arrays with the
                 weights and biases of each layer.
    -functions: [list] the activation functions of the
                 layers
    -X: [np.array] the data

    Returns: The output of the network given data X.
             Specifically the activation of the last layer.
    """
    L = len(parameters)
    A = X
    count = 0
    for i in range(0, L, 2):
        Z = np.dot(parameters[i], A) + parameters[i+1]
        A = functions[count].calculate(Z)
        count += 1
    return A


def compute_cost(A, Y):
    """Fourth helper function for gradient checking.
    Calculate the cost of the networks output.
    Inputs:
    -A: The last layers activation
    -Y: The true labels
    Returns:
    -costs: The cost of the networks output
    """
    m = Y.shape[1]
    logprobs = np.multiply(-np.log(A), Y) + np.multiply(-np.log(1 - A), 1 - Y)
    cost = 1./m * np.sum(logprobs)
    return cost


def gradient_checking(NeuralNetwork, X, Y, num_of_iterations=1, epsilon=1e-7):
    """performing gradient checking to a neural network
    Arguments:
        -NeuralNetwork: A NeuralNetwork object, on witch will be performed the gradient checking
        -X: The input layer of m sample units and n futures
        -Y: The labels for each sample unit
        -num_of_training_iterations: The number of iterations that the network will be trained before checking
                                      the grads
        -epsilon: tiny shift to the input to compute approximated gradient with formula

        Because we need the parameters before the last iteration of the training, where the grads will be calculated,
        instead of NeuralNetwork.train(X, Y, num_of_training_iterations) the algorithm will use
        NeuralNetwork.train(X, Y, 1) in a for loop, saving the parameters and then performing training
        (front and back prop)"""

    # Training the network for num_of_iterations
    Opt = GradientDescent(1, 0.01, None)
    for i in range(num_of_iterations):
        # terning the parameters and gradients to one vector
        parameter_vector = parameters_to_vectors(NeuralNetwork.parameters)
        Opt.batch_epoch(NeuralNetwork, X, Y)

    num_parameters = len(parameter_vector)
    # used to return the parameters from the vector: parameter_vector (see vector_to_parameters)
    parameters_dims = [NeuralNetwork.parameters[i].shape for i in range(len(NeuralNetwork.parameters))]
    # terning the gradients calculated after training to one vector
    grad = parameters_to_vectors(NeuralNetwork.grads)
    grads_dims = [NeuralNetwork.grads[i].shape for i in range(len(NeuralNetwork.grads))]

    layers_functions = [layer.function for layer in NeuralNetwork.Layers[1:]]
    # J_plus[i] will be the cost when to the i entry of the parameter_vector is added epsilon.
    # similar for J_minus.
    # gradapprox holds the values for the approximation of the derivatives with respect to parameter i
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))

    print("Gradients testing", end='')
    for i in range(num_parameters):
        theta_plus = np.copy(parameter_vector)
        theta_plus[i][0] += epsilon
        theta_minus = np.copy(parameter_vector)
        theta_minus[i][0] -= epsilon
        plus_parameters = vector_to_parameters(theta_plus, parameters_dims)

        minus_parameters = vector_to_parameters(theta_minus, grads_dims)
        # front propagation using the parameters theta_plus and theta minus
        # using theta plus
        AL_plus = Forward_calculations(X, plus_parameters, layers_functions)

        J_plus[i] = compute_cost(AL_plus, Y)
        # using theta minus
        AL_minus = Forward_calculations(X, minus_parameters, layers_functions)
        J_minus[i] = compute_cost(AL_minus, Y)
        # approximation of the gradient with respect to the i parameter
        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)
        print(".", end='')

    # Compare gradapprox to backward propagation gradients by computing difference.
    numerator = np.linalg.norm(grad - gradapprox)  # Step 1'
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)  # Step 2'
    difference = numerator / denominator

    if difference > 2*epsilon:
        print("\033[91m"+"FAILED"+"\033[0m")
        print("difference: "+str(difference)+" is bigger than 2x epsilon: "+str(epsilon))
    elif difference > epsilon:
        print("\033[93m" + "Inconclusive" + "\033[0m")
        print("difference: "+str(difference)+" is between epsilon and 2x epsilon: "+str(epsilon))
    else:
        print("\033[92m" + "PASSED" + "\033[0m")
        print("difference: "+str(difference)+" is smaller than epsilon: "+str(epsilon))
    print("============================================================================")
    return difference
