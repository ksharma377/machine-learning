# This file generates the training data for univariate linear regression.

# Generates: "../data/training_data.csv"
# Each row in the file represents [x, y]

# Input: x
# Parameters: w0, w1
# Output: y = w0 + (w1 * x)
# Number of training examples: m

import random

w0 = 34.715
w1 = -79.074
m = 10000
lower_limit = -100000
upper_limit = 100000

filename = "../data/training_data.csv"

print("\n*** Beginning training data generation ***\n")

with open(filename, "w") as fp:
	fp.write("x,y\n")
	for i in range(m):
		x = random.uniform(lower_limit, upper_limit)
		y = w0 + (w1 * x)
		fp.write("{},{}\n".format(x, y))

print("*** Trainig data generated successfully ***\n")
