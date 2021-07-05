import numpy as np 
from keras.datasets import mnist
from keras.utils import to_categorical

def relu(x):
	return np.maximum(0,x)

def softmax(x):
	max_ = np.max(x)
	return np.exp(x-max_)/np.sum(np.exp(x-max_))

def dReLu(x):
	data = np.copy(x)
	for i in range(x.shape[0]):
		for j in range(x.shape[1]):
			if(x[i][j] > 0):
				data[i][j] = 1
			else:
				data[i][j] = 0
	return data

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#need that channel dimension, normalized float32 tensor
X_train = train_images.reshape((60000, 28*28, 1)).astype('float32')/255 
Y_train =  to_categorical(train_labels).reshape((60000, 10, 1))
X_test = test_images.reshape((10000, 28*28, 1)).astype('float32')/255 
Y_test =  to_categorical(test_labels).reshape((10000, 10, 1))

x = X_train
y = Y_train

# layer 1
np.random.seed(23)
weights_1 = np.random.rand(128, 28*28) - 0.5
b_1 = np.random.rand(128, 1) - 0.5
		
# layer 2
weights_2 = np.random.rand(70, 128) - 0.5
b_2 = np.random.rand(70, 1) - 0.5

# output layer
weights_3 = np.random.rand(10, 70) - 0.5
b_3 = np.random.rand(10, 1) - 0.5


epochs = 40
BS = 128
lr = 0.0001
gamma = 0.95

losses = []
accuracy =[]
for i in range(epochs):
	if i >= 20:
		BS = 32
		gamma = 0.99
	acc = 0
	loss = 0
	count = 0
	np.random.seed(23)
	samp = np.random.randint(0, y.shape[0], size=y.shape[0]//BS)
	#samp = np.arange(0, y.shape[0], BS) #batch sample size
	for k in samp:
		#forward
		z_1 = np.dot(weights_1, x[k])+ b_1
		a_1 = relu(z_1) 
		z_2 = np.dot(weights_2, a_1) + b_2
		a_2 = relu(z_2)
		z_3 = np.dot(weights_3, a_2) + b_3
		a_3 = softmax(z_3)

		count += 1

		loss += ((a_3 - y[k])**2).mean()
		if (np.argmax(a_3) == np.argmax(y[k])):
			acc += 1
		#backward
		#output layer
		dc_dz_o = (a_3 - y[k])
		dz_dw_o = a_2.T
		dw3 = np.dot(dc_dz_o,dz_dw_o)/y[k].shape[0]
		db3 = dc_dz_o/y[k].shape[0]

		#layer 2
		dc_dz_t1 = dc_dz_o
		dz_t1_da = weights_3.T
		da_dz_2 = dReLu(z_2) 
		dc_dz_2 = np.dot(dz_t1_da, dc_dz_t1)* da_dz_2
		dz_dw_2 = a_1.T
		dw2 = np.dot(dc_dz_2, dz_dw_2)/y[k].shape[0]
		db2 = dc_dz_2/y[k].shape[0]

		#layer 1
		dc_dz_t1 = dc_dz_2
		dz_t1_da = weights_2.T
		da_dz_1 = dReLu(z_1) 
		dc_dz_1 = np.dot(dz_t1_da, dc_dz_t1)* da_dz_1
		dz_dw_1 = x[k].T
		dw1 = np.dot(dc_dz_1, dz_dw_1)/y[k].shape[0]
		db1 = dc_dz_1/y[k].shape[0]


		if count == 1:
			momentum = [lr*dw1, lr*db1, lr*dw2, lr*db2, lr*dw3, lr*db3]
		else:
			momentum = [gamma*momentum[0] + lr*dw1, gamma*momentum[1] + lr*db1, gamma*momentum[2] + lr*dw2,
						gamma*momentum[3] + lr*db2, gamma*momentum[4] + lr*dw3, gamma*momentum[5] + lr*db3]

		#sgd
		weights_1 -= momentum[0]
		b_1 -= momentum[1]
				
		weights_2 -= momentum[2]
		b_2 -= momentum[3]

		weights_3 -= momentum[4]
		b_3 -= momentum[5]

	acc /= count
	loss /= count
	accuracy.append(acc)
	losses.append(loss)
	#print(f'epoch: {i}, accuracy: {acc}')
	print(f'epoch: {i+1}, accuracy: {acc}, loss: {loss}')


print('Evaluating')
precision = 0
for k in range(len(Y_test)):
	z_1 = np.dot(weights_1, X_test[k])+ b_1
	a_1 = relu(z_1) 
	z_2 = np.dot(weights_2, a_1) + b_2
	a_2 = relu(z_2)
	z_3 = np.dot(weights_3, a_2) + b_3
	a_3 = softmax(z_3)

	if (np.argmax(a_3) == np.argmax(Y_test[k])):
		precision += 1

print(f'Network got {precision/len(Y_test)} right')


import matplotlib.pyplot as plt
plt.plot(range(epochs), accuracy, label='accuracy')
plt.plot(range(epochs), losses, label='loss')
plt.title('Trainning results')
plt.legend()
plt.show()