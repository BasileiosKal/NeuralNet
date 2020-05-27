class LayersBaseClass:
    """Base class for the layers objects.
    It sets the name and Regularization type
    as well as the dimensions of a layer.

    The dimensions will be a dictionary with the
    dimensions of the layer. For example for a
    FC Layer will be something like: {"in": 5, "out": 10}

    TODO: add assertions in __init__ for the dimensions

    """
    def __init__(self, *args, **kwargs):
        dims_dict = args[0]
        self.dimensions = dims_dict
        self.name = kwargs["name"]
        self.Regularization = kwargs["Regularization"]


class NeuralNetworkBase:
    """Base neural network class. The Base class is necessary as to
    be able to extent to different network architectures with the same
    main classes, namely front_propagation, train and predict.

    The network class below (NeuralNetwork) supports only Convolutional and
    Planar networks, depending on the type of layers that will be passed in.
    RNNs have a very different structure and so a different object will be
    created, that will extent the NeuralNetworkBase, to implement RNNs.

    """
    def __init__(self, layers_dim):
        # self.layers_dim = layers_dim
        L = len(layers_dim)
        self.L = L

    def front_propagation(self, *args, **kwargs):
        raise NotImplementedError

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        raise NotImplementedError
