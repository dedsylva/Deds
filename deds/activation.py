import numpy as np

def ReLu(x):
    return np.maximum(0,x)

def dReLu(x):
    data = np.array(x, copy=True)
    data[x<= 0] = 0
    return data

def Sigmoid(x):
    return 1/(1 + np.exp(-x))

def dSigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def Linear(x):
    return x

def dLinear(x):
    x.fill(1)
    return x

def Softmax(x):
    max_ = np.max(x)
    return np.exp(x-max_)/sum(np.exp(x-max_))

def dSoftmax(x):
    return x*(1-x)

def Tanh(x):
    return np.tanh(x)

def dTanh(x):
    return 1 - np.square(Tanh(x))
