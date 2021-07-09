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

# Adam (Adaptive Moment Estimation): the greates of them all.
# A simple combination of RMSProp + Moentum approach
def Adam(model, gradients, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
	t = 0
	m_w = [np.zeros_like(model[i][0]) for i in range(len(model))]
	v_w = [np.zeros_like(model[i][0]) for i in range(len(model))]
	m_b = [np.zeros_like(model[i][1]) for i in range(len(model))]
	v_b = [np.zeros_like(model[i][1]) for i in range(len(model))]
	for i in range(len(model)):
		t+=1
		m_w[i] = b1 * m_w[i] + (1 - b1) * gradients[-1-i][1]
		v_w[i] = b2 * v_w[i] + (1 - b2) * np.square(gradients[-1-i][1])
		m_b[i] = b1 * m_b[i] + (1 - b1) * gradients[-1-i][2]
		v_b[i] = b2 * v_b[i] + (1 - b2) * np.square(gradients[-1-i][2])
		mhat_w = m_w[i] / (1. - b1**t)
		vhat_w = v_w[i] / (1. - b2**t)
		mhat_b = m_b[i] / (1. - b1**t)
		vhat_b = v_b[i] / (1. - b2**t)
		model[i][0] -= lr*mhat_w / (np.sqrt(vhat_w) + eps) #updating weights
		model[i][1] -= lr*mhat_b / (np.sqrt(vhat_b) + eps) #updating biases
	return model

# RMSProp and Momentum take contrasting approaches. 
# While momentum accelerates our search in direction of minima, 
# RMSProp impedes our search in direction of oscillations.
# That's why RMSProp doesn't have momentum
def RMSProp(model, gradients, lr=0.001, b1=0.9, eps=1e-7):
	m_w = [np.zeros_like(model[i][0]) for i in range(len(model))]
	m_b = [np.zeros_like(model[i][1]) for i in range(len(model))]
	for i in range(len(model)):
		m_w[i] = b1 * m_w[i] + (1 - b1) * np.square(gradients[-1-i][1])
		m_b[i] = b1 * m_b[i] + (1 - b1) * np.square(gradients[-1-i][2])
		model[i][0] -= lr*gradients[-1-i][1] / (np.sqrt(m_w[i]) + eps) #updating weights
		model[i][1] -= lr*gradients[-1-i][2] / (np.sqrt(m_b[i]) + eps) #updating biases
	return model	