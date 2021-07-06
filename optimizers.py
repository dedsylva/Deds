import numpy as np

def SGD(model, gradients, momentum):
	if momentum:
		k = 0
	else:
		k = 1

	for i in range(len(model)):
		model[i][0] -= gradients[-1-i][0+k] #updating weights
		model[i][1] -= gradients[-1-i][1+k] #updating biases

	return model
