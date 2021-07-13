import sys
import numpy as np
from model import Model
import pandas as pd
import math
from database import Wheat, MNIST

def main(argv):
	data = [a.split('=') for a in argv]

	if data[0][0] == 'model':
		if data[0][1] == 'MNIST':
			db = MNIST()
			X_train, X_test, Y_train, Y_test = db.get_data()
			epochs = 80
			BS = 128
			lr = 0.001
			gamma = 0.95

			NN = Model()

			model = NN.Input(128, input_shape=X_train.shape[1], activation='ReLu')
			model = NN.Dense(128, 70, model, activation='ReLu', regularization='l2', reg=0.0001)
			model = NN.Dropout(model, p=0.33)
			model = NN.Output(70, 10, model, activation='Softmax')

			#compile model
			NN.Compile(optimizer='SGD', loss='MSE', metrics='accuracy', lr=lr, 
					   momentum=True, gamma=gamma)

			#train the model
			model, loss, accuracy = NN.Train(model, X_train, Y_train, 
				epochs=epochs, batch=BS, categoric=True)

			#evaluate the network
			precision = NN.Evaluate(model, X_test, Y_test, True)

			import matplotlib.pyplot as plt
			plt.plot(range(epochs), accuracy, label='accuracy')
			plt.plot(range(epochs), loss, label='loss')
			plt.title('Trainning results')
			plt.legend()
			plt.show()

		elif data[0][1] == 'Wheat':
			db = Wheat()
			train = float(data[1][1]) if len(data) > 1 else 0.9
			X_train, X_test, Y_train, Y_test = db.get_data(train=train)
			epochs = 10000
			BS = 8
			NN = Model()

			model = NN.Input(10, input_shape=X_train.shape[1], activation='ReLu')
			model = NN.Dense(10, 5, model, activation='ReLu')
			model = NN.Output(5, 1, model, activation='Linear')

			#train the model
			model, loss, accuracy = NN.Train(model, X_train, Y_train, 
				loss='MSE', opt='SGD', epochs=epochs, batch=BS, categoric=False, lr=0.0001)	

			#evaluate the network
			precision = NN.Evaluate(model, X_test, Y_test, False)

			import matplotlib.pyplot as plt
			plt.plot(range(epochs), accuracy, label='accuracy')
			plt.plot(range(epochs), loss, label='loss')
			plt.title('Trainning results')
			plt.legend()
			plt.show()

		else:
			raise Exception ('The Model you entered are not available')
	else:
		raise ValueError ('The argument \'{}\' is invalid!'.format(data[0][0]))

if __name__ == '__main__':
	main(sys.argv[1:])