import numpy as np
from Utilities.Ploting import plot_decision_boundary
import matplotlib.pyplot as plt
import copy
from Networks.BaseClasses import NeuralNetworkBase, LayersBaseClass


class InputLayer(LayersBaseClass):
    """The input layer of a network. Its "activation"
    is set to be the data that will be used as input
    for the network, in order to be able to be used
    for propagation, where the activation of a layer
    depends on the activation of a layer depends on
    the activation of the previous one.

    """
    def __init__(self, dim):
        super().__init__({"out": dim}, name="Input Layer", Regularization=None)
        self._activation = None

    @property
    def activation(self):
        return self._activation

    @activation.setter
    def activation(self, X):
        assert X.shape[0] == self.dimensions["out"]
        self._activation = X


class FCLayer(LayersBaseClass):
    """Foully Connected Layers class.
    Inputs: size: [int] The size of the layer.
            function: [Function object] The activation function.
            name: [str] The name of the layer.

    ==============================================================
    Important: The class is not meant to be used directly.
               A FCLayer object must first be built using the Builder
               class (FCLayerBuilder) that follows.
    ==============================================================

    The layers main function is to calculate its activation and
    the gradients of its parameters when propagation passes through
    the layer.

    """
    def __init__(self, size, function, Regularization, name):
        super().__init__({"in": None, "out": size}, name=name, Regularization=Regularization)
        self.function = function
        self.v_parameter = {}

    # Forward calculations
    def forward_calc(self, value):
        """Given a value calculate the activation of the
        layer with that value as the input.

        During forward propagation the value will be
        the activation of the previous layer in the network.

        """
        if not self.W.any():
            raise ValueError("Parameters of layer {} are not initialized".format(self.name))

        self.linear_z = np.dot(self.W, value) + self.b
        self.activation = self.function.calculate(self.linear_z)

    # backward calculation
    def backward_calc(self, prev_layer, m):
        """given a layer connected to that one
        calculate the gradients ot the cost function
        with respect to the parameters of the layer

        During back propagation the prev_layer will
        be the previous layer in the network.

        """
        assert prev_layer.activation.shape == (self.dimensions["in"], m)

        self.dW = (1 / m) * np.dot(self.dZ, prev_layer.activation.T)
        self.db = (1 / m) * np.sum(self.dZ, axis=1, keepdims=True)

        assert self.dW.shape == self.W.shape
        assert self.db.shape == self.b.shape


class FCLayerBuilder:
    """Builder for the FCLayer class. It contains
    two different methods for building a FCLayer
    object: connect_to and set_input. Both initialize
    the layers input dimension and use rand_initialize
    to initialize the weights and biases accordingly.

    connect_to: Takes another layer object and builds
                the FCLayer with dimensions to fit.
    set_input: Takes as argument the layers input
               dimension and builds the FCLayer object.
    =======
    Example:
    >>> layer_parameters = {"size": 5,
                            "activation": relu,
                            "Regularization": None
                            "name": "FC Layer"}
    >>> previous_layer_parameters = {"size": 10,
                                     "activation": relu,
                                     "Regularization": None
                                     "name": "FC Layer"}

    >>> layer_parameters = list(layer_parameters.values())
    >>> previous_layer_parameters = list(layer_parameters.values())

        # Let previous_layer to have size of 10 and input size of 2.
    >>> previous_layer =  FCLayerBuilder(*previous_layer_parameters).set_input(2).build()
        # Using the connect_to function
    >>> Layer1 = FCLayerBuilder(*layer_parameters).connect_to(previous_layer).build()
        # Using the set_input function
    >>> Layer2 = FCLayerBuilder(*layer_parameters).set_input(10).build()

    Output:
    >>> Layer1.dimensions
    {'in': 10, 'out': 5}
    >>> Layer1.W.shape
    (5, 10)
    >>> Layer2.dimensions
    {'in': 10, 'out': 5}
    >>> Layer2.W.shape
    (5, 10)

    Both the approaches will create a FCLayer object with input dimensions 10 and size 5
    and initialize weights with shape (5, 10) and biasses with shape (5, 1).

    ==============
    Note:
    Upon creation the activation and gradient parameters are initialized
    to None. This is for readability reasons so it will be apparent what
    attributes the class contains. The parameters are actually initialized
    when forward or backward propagation passes through the layer. More
    specifically when the forward_calc and backward_calc functions are used.

    """
    def __init__(self, size, function, Regularization, name):
        self.FCLayer = FCLayer(size, function, Regularization, name)
        # activation parameters
        self.FCLayer.linear_z = None
        self.FCLayer.activation = None
        # Gradients
        self.FCLayer.dW = None
        self.FCLayer.db = None
        self.FCLayer.dZ = None

    def rand_initialize(self, constant):
        """initializing the weights and bias of the layer.
        A random vector is generated and multiplied with
        a close to zero constant. If the constant is not
        specified is set automatically.

        """
        if constant:
            mul_parameter = constant  # mul_parameter: the parameter witch the weights will be multiplied
        else:
            mul_parameter = np.sqrt(2 / self.FCLayer.dimensions["out"])

        W = np.random.randn(self.FCLayer.dimensions["out"], self.FCLayer.dimensions["in"]) * mul_parameter
        b = np.zeros((self.FCLayer.dimensions["out"], 1))
        return W, b

    def connect_to(self, other_layer, init_constant=None):
        """Sets the dimension of the input of the layer
        to be the output dimension of another layer and
        initializes the weights and biases accordingly

        """
        self.FCLayer.dimensions["in"] = other_layer.dimensions["out"]
        W, b = self.rand_initialize(init_constant)
        self.FCLayer.W = W
        self.FCLayer.b = b
        return self

    def set_input(self, input_size, init_constant=None):
        """Sets the dimension of the input of the layer
        to be equal to the input_size and initializes
        the weights and biases accordingly

        """
        self.FCLayer.dimensions["in"] = input_size
        W, b = self.rand_initialize(init_constant)
        self.FCLayer.W = W
        self.FCLayer.b = b
        return self

    def build(self):
        return self.FCLayer


# ================================================================================================== #
#                                          Neural Network                                            #
# ================================================================================================== #
class NeuralNetwork(NeuralNetworkBase):
    """Neural network class for Planar and Convolutional networks.

    Inputs: layers: [List] Its element of the list will be a dictionary which will hold the
                           parameters of the layers. For FC layers the elements of the dictionary
                           must be the type, dim, activation (the activation function) and
                           regularization. The first element of the layers list must be a dictionary
                           with the input dimensions.
    ==================================
    Example:

         >>>Input = {"type": InputLayer,
                     "dim": X.shape[0]}

        >>> FClayer1 = {"Builder": FCLayerBuilder,
                        "dim": 5,
                        "activation": relu,
                        "Regularization": None,
                        "name": "FC Layer 1"}

        >>> FClayer2 = {"Builder": FCLayerBuilder,
                        "dim": 4,
                        "activation": relu,
                        "Regularization": None,
                        "name": "FC Layer 2"}

        >>> FClayer3 = {"Builder": FCLayerBuilder,
                        "dim": 2,
                        "activation": sigmoid,
                        "Regularization": None,
                        "name": "FC Layer 3"}

        >>> layers = [Input, FClayer1, FClayer2, FClayer3]

        >>> Network = NeuralNetwork(layers)
    ==================================
    This will create a neural network with the above layers. To train the network
    you can create an optimizer and pass it to the train function.
    ==================================
    Example:
        >>> Hyper_parameters = {"iterations": 1000,
                                "learning_rate": 0.01,
                                "mini_batch": 64}
        >>> Hyper_parameters = Hyper_parameters.values()

        # Create an optimizer that uses Gradient Descent
        >>> Optimizer = GradientDescent(*Hyper_parameters)

        # Train the Network on data X with labels Y.
        >>> cost = Network.train(X, Y, Optimizer, plot_boundary=True, plot_cost=True)

    ==================================
    Results:
        --------------------------------------------------------------------------------
        Gradient with momentum. Cost after 1000 iterations: 0.004138054247924641
        Accuracy with Momentum: 0.876
        --------------------------------------------------------------------------------

        """
    def __init__(self, layers):
        super().__init__(layers)
        self.Layers = [InputLayer(layers[0]["dim"])]
        previous_layer = self.Layers[0]

        for index, layer in enumerate(layers[1:]):
            Builder = layer["Builder"]
            layer_parameters = list(layer.values())

            Layer = Builder(*layer_parameters[1:]).connect_to(previous_layer).build()

            self.Layers.append(Layer)
            previous_layer = Layer

    @property
    def parameters(self):
        parameters = []
        for layer in range(1, self.L - 1):  # keys from 1 to L-1 (not counting layer 0)
            parameters.append(self.Layers[layer].W)
            parameters.append(self.Layers[layer].b)
        return parameters

    @property
    def grads(self):
        grads = []
        for layer in range(1, self.L - 1):  # keys from 1 to L (not counting layer 0)
            grads.append(self.Layers[layer].dW)
            grads.append(self.Layers[layer].db)
        return grads

    def train(self, X, Y, Optimization_algorithm, plot_boundary=False, plot_cost=False, print_result=True):
        """Function for training the network, using the
        Optimize method of the Optimization_algorithm.

        Inputs:
            -X: [np.array] The training data.
            -Y: [np.array] The true labels.
            -Optimization_algorithm: [Optimizer object] The optimization algorithm used for training.
            -plot_boundary: [boolean] If True plotting the decision boundary if true and if able.
            -plot_cost: [boolean] If True plotting the cost over the iterations.
            -print_result: [boolean] If true printing the accuracy and the cost at the end of the training.

        Returns:
            -cost [list]: A list with the costs on every iteration.

            """
        cost = Optimization_algorithm.Optimize(self, X, Y, print_result)
        self.plotting(X, Y, cost, plot_boundary, plot_cost, Optimization_algorithm.AlgorithmType)

        return cost

    def plotting(self, X, Y, cost, plot_boundary, plot_cost, title):
        if plot_boundary or plot_cost:
            size = 6  # the window size for each subplot.
            # the figure window size will depend on what the user wants to plot.
            fig = plt.figure(figsize=(size * plot_boundary + size * plot_cost, 5))
        subplot_n_of_cols = plot_boundary + plot_cost

        if plot_boundary:
            plt.subplot(1, subplot_n_of_cols, 1)
            plt.title(title + ": Decision Boundary")
            axes = plt.gca()
            axes.set_xlim([-1.5, 2.5])
            axes.set_ylim([-1, 1.5])
            plot_decision_boundary(lambda x: self.predict(x.T), X, Y)

        if plot_cost:
            plt.subplot(1, subplot_n_of_cols, subplot_n_of_cols)
            plt.title(title + ": Cost")
            plt.plot(cost)
        plt.show()

    def front_propagation(self, X):
        """Helper function for calculating the predictions
        of the network. It performs forward propagation
        given input data X

        """
        A_prev = X
        for layer in self.Layers[1:]:
            current_layer = layer
            current_layer.forward_calc(A_prev)
            A_prev = current_layer.activation
        return A_prev

    def predict(self, X, threshold=0.5):
        """Gives the prediction after calculating
        the output of the network, given input data X
        by compering the output to the threshold

        """
        aL = self.front_propagation(X)
        predictions = (aL > threshold)
        return predictions

    def reset(self, initial_parameters=None):
        """Function that resets all the gradients
        of all the layers to None and there parameters
        ether to None or to the ones specified by
        initial_parameters.

        Inputs:
        -initial_parameters: [list] A list with the weighs and
                                    biases of each layer.

        Note: initial_parameters must be of the form
            [W1, b1, W2, b2,.....]  where W1 and b1 will become the
            weight and bias of the first layer, W2, b2 the weight and bias
            of the second layer and so on.

            """
        for layer in self.Layers:
            layer.dZ = None
            layer.db = None
            layer.dW = None
            layer.v_parameter = {}

        if initial_parameters is not None:
            for layer, init_W, init_b in zip(self.Layers[1:], initial_parameters[0:-1:2],
                                             initial_parameters[1:len(initial_parameters):2]):
                layer.W = copy.copy(init_W)
                layer.b = copy.copy(init_b)
        else:
            for layer in self.Layers:
                layer.rand_initialize(None)
