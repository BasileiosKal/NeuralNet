# NeuralNet

A deep learning framework built from scratch using Python.

## Overview

This project is meant to be a full implementation of Neural Networks, with everything from planar networks to a deep 
Convolutional/ Residual neural networks and sequence models, built entirely from scratch, using only Python and numpy. I mainly
developed this project for reserch purposes and to get a deeper understading of the deep learning algorithms and practices by
coding them my self from scratch. 

## Structure 

The architecture of the framework is similar to that of TensorFlow. In sorts there are three main classes: the Layers class, the Networks class and the Optimizers class.
- ##### The Layers class:
  Contains variables for the parameters and the gradients as well as functions for forward and backward calculations, that will be used when forward or backward propagation passes through the layer. Currently only fully connected layers are supported.
  
- ##### The Networks class:
  Contains variables for the hyperparameters of the network architecture and upon creation it initializes the layers specified by this hyperparameters and stores them to a list. It also contains functions for forward propagation, training(this will will use the optimizers class), predicting, ploting the cost or the decision boundary (if able) etc.
  
- ##### The Optimizer class:
   This class is responsible for the training of the network and mainly for back propagation. Contains variables for the hyperparameters associated with this task like the learning rate, mini batch size or regularization type. It also carrys different functions for executing back propagation depenting on this hyper parameters. For example there deferent functions for mini batch gradient dechent and batch gradient dechent and the algorithm will choose one based on the value of the mini_batch parameter (batch if set to None, mini batch otherwise).

Besides this classes there other functionalities as well, like a Functions class for the activation functions and their derivatives, functions for loading and ploting data, a parallel train function used to train multiple networks at the same time efficiently etc.


## Examples 
#### Training a BinaryClassifier on the Moons data set:
![Example 1](https://raw.githubusercontent.com/BasileiosKal/NeuralNet/master/Images/example1.png)
#### Training multiple networks using different optimizers for each one.
![Example 2](https://raw.githubusercontent.com/BasileiosKal/NeuralNet/master/Images/example2.png)


## Future work
As it is, the module contains only fully conected layers and so curently i'm working on adding convolutional layers, ResNets, inception layers exetra. Also the optimizers that are implemented at the time are Gradient Dechent, Momentum Gradient Dechent and RMSprop, so i'm working to add the Adam optimizer to the mix.
