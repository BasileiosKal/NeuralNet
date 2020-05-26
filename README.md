# NeuralNet

A deep learning framework built from scratch using Python and optimized with Cython.

## Overview

This project is meant to be a full implementation of Neural Networks, with everything from planar networks to a deep 
Convolutional/ Residual neural networks and sequence models, built entirely from scratch, using only Python and numpy. I mainly
developed this project for research purposes and to get a deeper understanding of the deep learning algorithms and practices by
coding them my self from scratch. 

## Structure 

The architecture of the framework is similar to that of TensorFlow. In sorts there are three main classes: the Layers class, the Networks class and the Optimizers class.
- ##### The Layers class:
  Contains variables for the parameters and the gradients as well as functions for forward and backward calculations, that will be used when forward propagation passes through the layer. The layers class uses a Builder design pattern to make the code cleaner. That means that every layer must first be build before it is used. Currently only fully connected layers are supported.
  
- ##### The Networks class:
  Contains variables for the hyperparameters of the network architecture and upon creation it initializes the layers specified by this hyperparameters and stores them to a list. It also contains functions for forward propagation, training(this will will use the optimizers class), predicting, plotting the cost or the decision boundary (if able) etc.
  
- ##### The Optimizer class:
   This class is responsible for the training of the network and mainly for back propagation. Contains variables for the hyperparameters associated with this task like the learning rate, mini batch size or regularization type. It also carries different functions for executing back propagation depending on this hyper parameters. For example there different functions for mini batch gradient descent and batch gradient descent and the algorithm will choose one based on the value of the mini_batch parameter (batch if set to None, mini batch otherwise).

Besides this classes there other functionalities as well, like a Functions class for the activation functions and their derivatives, functions for loading and ploting data, a parallel train function used to train multiple networks at the same time efficiently etc.


## Examples 
#### Setting the structure of the Network:
![Example 1](https://raw.githubusercontent.com/BasileiosKal/NeuralNet/master/Images/example1.png)
#### Build an gradient descent optimizer.
![Example 1](https://raw.githubusercontent.com/BasileiosKal/NeuralNet/master/Images/Example2.png)
#### Create the Network and train it with the optimizer created before.
![Example 2](https://raw.githubusercontent.com/BasileiosKal/NeuralNet/master/Images/Example3.png)
#### Results
![Resulting graphs](https://raw.githubusercontent.com/BasileiosKal/NeuralNet/master/Images/Figure_1.png)
![Overall results](https://raw.githubusercontent.com/BasileiosKal/NeuralNet/master/Images/ResultsExample.png)

## Testing
For testing i used the unittest module. Each package (exept Utilitis) contains a Testing subfolder with the test for those specific modules. In order to run all the test at once the user can run the Testing/run_tests.py module.
Alternatively each testing module can be ran by it self using the -m unittest [testing module path]. 

## Cython
The OptimizationAlgorithms module has a Cython alternative (called OptimizationAlgorithms_Cython.pyx) that can run almost 30 times faster in comparison to the Python one. The reazon i have kept both pure Python and Cython modules is that the Cython optimized gradient descent seems to run slower for large numper of iterations (more than 10000). Until this issue is fixed i decided to have Cython as an alternative rather than use it as the main module.

## Future work
As it is, the module contains only fully connected layers and so currently i'm working on adding convolutional layers, ResNets, inception layers exc. Also the optimizers that are implemented at the time are Gradient Descent, Momentum Gradient Descent and RMSprop, so i'm working to add the Adam optimizer to the mix.
