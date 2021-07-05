import numpy as np
from model import Model
import pandas as pd
import math
from database import Wheat, MNIST


# model is a list where:
# len(model) = number of layers (counting the input and output)
# model[i][0] is the Weights matrix of layer i
# model[i][1] is the bias vector of layer i
# model[i][2] is the activation of the layer i <<-- limitation: currently we can only have the same activation for the entire layer

db = Wheat()
X_train, X_test, Y_train, Y_test = db.get_data()

NN = Model()

model = NN.Input(10, input_shape=X_train.shape[1], activation='ReLu')
model = NN.Dense(10, 7, model, activation='ReLu')
model = NN.Dense(7, 5, model, activation='ReLu')
model = NN.Output(5, 1, model, activation='Linear')

#train the model
loss, accuracy = NN.Train(model, X_train, Y_train, 
	loss='MSE', opt='SGD', epochs=10, batch=8, categoric=False, lr=0.0005)
