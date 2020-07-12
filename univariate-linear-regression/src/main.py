"""
This is the main file which builds the univariate linear
regression model.

Training data filename: "training_data.csv"
Training data path: "../data/"

The model trains on the above data and outputs the parameters and the accuracy.
The number of training cycles is controlled by the variable "epochs".

Input: x
Parameters: w0, w1
Output: y
Heuristic: h(x) = w0 + (w1 * x)
Number of training examples: m
Batch size: b
Learning rate: r
Cost function = MSSE (Mean sum of squares of errors)
	C(w0, w1) = (1 / 2m) sigma((h(x) - y)^2)
Optimizer algorithm: Batch Gradient Descent
"""

import random

"""
Initializes the parameters
"""
def initialize_parameters():
	global w0, w1, r, epochs
	w0 = random.random()
	w1 = random.random()
	r = 0.1
	epochs = 100

initialize_parameters()
