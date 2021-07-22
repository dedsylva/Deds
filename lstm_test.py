import numpy as np

def sigmoid(x):
	return 1/(1 + np.exp(-x))

def tanh(x):
	return np.tanh(x)	
## https://www.youtube.com/watch?v=9zhrxE5PQgY&ab_channel=SirajRaval	
## another ref: https://github.com/nicoladaoud/RNN-LSTM-from-scratch/blob/master/RNN-LSTM.py
## https://www.kaggle.com/navjindervirdee/lstm-neural-network-from-scratch
## https://machinelearningmastery.com/gentle-introduction-backpropagation-time/