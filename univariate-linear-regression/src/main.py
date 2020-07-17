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
Cost function = MSE (Mean squared error)
	C(w0, w1) = (1 / 2m) sigma((h(x) - y)^2)
Optimizer algorithm: Batch Gradient Descent
"""

import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

"""
Initializes the parameters
"""
def initialize_parameters():
	global w0, w1, r, epochs
	w0 = random.random()
	w1 = random.random()
	r = 0.1
	epochs = 100

"""
Reads the data in csv format and splits into training and testing data (80-20)
"""
def read_data():
	df = pd.read_csv('../data/training_data.csv')
	global train_data, test_data
	train_data, test_data = train_test_split(df, test_size = 0.2)

def calculate_mse(batch):
	m = len(batch)
	error = 0
	for row in batch.itertuples():
		x, y = row.x, row.y
		h = w0 + (w1 * x)
		error += (h - y) ** 2
	return error / (2 * m)

"""
Trains the linear regression model.
1. Sample a batch of fixed size.
2. Calculate the error for this batch.
3. Update the parameters using Gradient Descent.
4. Repeat for epochs.
"""
def train_model():
	batch_size = 1000
	for epoch in range(epochs):
		batch = train_data.sample(n = batch_size)
		error = calculate_mse(batch)
		print("Epoch: {}, Error: {}".format(epoch, error))

if __name__ == "__main__":
	read_data()
	initialize_parameters()
	train_model()
