import setuptools
import sys
import os
import numpy
from Cython.Build import cythonize

with open("README.md", "r") as file:
    long_description = file.read()

this_file = os.path.abspath(__file__)
NeuralNet_dir = os.path.join(os.path.dirname(this_file), "..")
NeuralNet_dir = os.path.normpath(NeuralNet_dir)
sys.path.insert(0, NeuralNet_dir)

sys.path.append(os.path.join(NeuralNet_dir, 'Optimization'))

setuptools.setup(
    name="example-pkg-YOUR-USERNAME-HERE",
    version="0.0.1",
    author="Vasilis Kalos",
    author_email="kalosbasileios@gmail.com",
    description="A Machine Learning package, build from scratch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BasileiosKal/NeuralNet",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    include_dirs = [numpy.get_include()],
    package_dir={'CythonTest': ''},
    ext_modules=cythonize(["CythonTest/OptimizationAlgorithms_Cython.pyx"])
)
