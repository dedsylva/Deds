import numpy as np

def MSE(x, y):
	return (x-y)**2

def dMSE(x,y):
	return 2*(x-y)

def Categorical_Cross_Entropy(x, y):
	return -x*np.log(y)

def dCategorical_Cross_Entropy(x, y):
	return x-y