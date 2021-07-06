import numpy as np

def SGD(model, gradients, momentum, l1=False, lambd_l1=0.00001,
		l2=False, lambd_l2=0.0001):
	if momentum:
		k = 0
	else:
		k = 1

	if l1:
		for i in range(len(model)):
			model[i][0] -= (gradients[-1-i][0+k] - lambd_l1) #updating weights
			model[i][1] -= gradients[-1-i][1+k] #updating biases
			return model
	elif l2:
		for i in range(len(model)):
			model[i][0] -= (gradients[-1-i][0+k] + lambd_l2*model[i][0]) #updating weights
			model[i][1] -= gradients[-1-i][1+k] #updating biases
			return model
	else:
		for i in range(len(model)):
			model[i][0] -= gradients[-1-i][0+k] #updating weights
			model[i][1] -= gradients[-1-i][1+k] #updating biases

	return model
