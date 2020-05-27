import data.Data_libraries.moons as data
from NeuralNet.Optimization.OptimizationAlgorithms_Py import GradientDescent
from NeuralNet.Networks.NeuralNetworks import NeuralNetwork, FCLayerBuilder, InputLayer
from NeuralNet.Utilities.Functions import RELU, Sigmoid

relu = RELU()
sigmoid = Sigmoid()

X, Y = data.load_moon_data(n_samples=1000)


Input = {"type": InputLayer,
         "dim": X.shape[0]}

FClayer1 = {"Builder": FCLayerBuilder,
            "dim": 5,
            "activation": relu,
            "Regularization": None,
            "name": "FC Layer 1"}

FClayer2 = {"Builder": FCLayerBuilder,
            "dim": 4,
            "activation": relu,
            "Regularization": None,
            "name": "FC Layer 2"}

FClayer3 = {"Builder": FCLayerBuilder,
            "dim": 2,
            "activation": relu,
            "Regularization": None,
            "name": "FC Layer 3"}

FClayer4 = {"Builder": FCLayerBuilder,
            "dim": 1,
            "activation": sigmoid,
            "Regularization": None,
            "name": "FC Layer 4"}

layers = [Input, FClayer4]



Hyper = {"iterations": 1000,
         "learning_rate": 0.01,
         "mini_batch": None}

Hyper_parameters = Hyper.values()

Network_py = NeuralNetwork(layers)

Optimizer_py = GradientDescent(*Hyper_parameters)

#=====================================================================================#

#start_py_time = time.time()

def run_p():
    Network_py.train(X, Y, Optimizer_py, plot_boundary=False, plot_cost=False, print_result=False)

run_p()
#py_time = (time.time() - start_py_time)
#print(py_time)

# diff = gradient_checking(Network, X, Y, num_of_iterations=10000)
# print("--- %s seconds ---" % (time.time() - start_time))