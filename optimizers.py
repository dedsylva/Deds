import numpy as np

def SGD(model, gradients, momentum):
	if momentum:
		k = 0
	else:
		k = 1

	for i in range(len(model)):
		if (model[i][3] == 'l1'): #only layers with l1 regularization
			ind = np.where(gradients[-1-i][0+k] - model[i][4] == 0) #find where the weights should be updated to zero
			l, c = ind #index of the line, index of the colum
			model[i][0] -= (gradients[-1-i][0+k] - model[i][4]) #updating weights
			model[i][1] -= gradients[-1-i][1+k] #updating biases
			for j in range(len(l)):
				weight_i = model[i][0] #weights of layer i
				weight_i[l[i],c[i]] = 0 #explicitly putting zeros where it should be (making l1 stable)

		elif (model[i][3] == 'l2'):
			model[i][0] -= (gradients[-1-i][0+k] + model[i][4]*model[i][0]) #updating weights
			model[i][1] -= gradients[-1-i][1+k] #updating biases

		else:
			model[i][0] -= gradients[-1-i][0+k] #updating weights
			model[i][1] -= gradients[-1-i][1+k] #updating biases

	return model
