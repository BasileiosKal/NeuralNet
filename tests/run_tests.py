import sys
import os
import unittest

this_file = os.path.abspath(__file__)
NeuralNet_dir = os.path.join(os.path.dirname(this_file), "..")
NeuralNet_dir = os.path.normpath(NeuralNet_dir)
sys.path.insert(0, NeuralNet_dir)

sys.path.append(os.path.join(NeuralNet_dir, 'Optimization/Tests'))
# from Optimization import testing_OptimizationAlgorithms

testing_modules = ['Networks/Tests', 'Optimization/Tests']

loader = unittest.TestLoader()
testSuite = loader.discover(testing_modules[0])

for module in testing_modules[1:]:
    loader = unittest.TestLoader()
    testSuite2 = loader.discover(module)
    testSuite.addTests(testSuite2)

testRunner = unittest.TextTestRunner(descriptions=False, verbosity=3)
testRunner.run(testSuite)
