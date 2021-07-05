import numpy as np

def SGD(model, gradients):
	for i in range(len(model)):
		model[i][0] -= gradients[-1-i][0] #updating weights
		model[i][1] -= gradients[-1-i][1] #updating biases

	return model
