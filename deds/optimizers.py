import numpy as np
from deds.utils.enumerators import Types, Regs

def SGD(model, gradients, momentum, lr):
  _types = [model[j][5] for j in range(len(model))]
  if momentum:
    offset  = 0 #Those offsets are dependencies of how we made the gradients. It's awful, I know, but it works for now.
    lr=1
  elif np.any(_types == Types.RNN):
    offset  = 0 
  else:
    offset  = 1

  for i in range(len(model)):
    if (model[i][3] == Regs.L1): #only layers with l1 regularization
      ind = np.where(gradients[-1-i][0+offset] - model[i][4] == 0) #find where the weights should be updated to zero
      l, c = ind #index of the line, index of the colum
      model[i][0] -= (lr*gradients[-1-i][0+offset] - model[i][4]) #updating weights
      model[i][1] -= lr*gradients[-1-i][1+offset] #updating biases
      for j in range(len(l)): #weights of layer i
        model[i][0][l[j],c[j]] = 0 #explicitly putting zeros where it should be (making l1 stable)

    elif (model[i][3] == Regs.L2):
      model[i][0] -= (lr*gradients[-1-i][0+offset] + model[i][4]*model[i][0]) #updating weights
      model[i][1] -= lr*gradients[-1-i][1+offset] #updating biases

    else:
      model[i][0] -= lr*gradients[-1-i][0+offset] #updating weights
      model[i][1] -= lr*gradients[-1-i][1+offset] #updating biases

  return model


def RNN_SGD(model, gradients, momentum, lr):
  _types = [model[j][5] for j in range(len(model))]

  for i in range(len(model)):
      model[i][0] -= lr*gradients[i][0] #updating weights
      model[i][1] -= lr*gradients[i][1] #updating biases

  return model

def RNN_Adam(model, gradients, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
  _types = [model[j][5].name for j in range(len(model))]
  # Adam doesn't have momentum (it estimates momentum), so we don't have to worry about that case

  t, j = 0, 0
  m_w = [np.zeros_like(model[i][0]) for i in range(len(model))]
  v_w = [np.zeros_like(model[i][0]) for i in range(len(model))]
  m_b = [np.zeros_like(model[i][1]) for i in range(len(model))]
  v_b = [np.zeros_like(model[i][1]) for i in range(len(model))]

  # Special case for RNN
  _indexes = [t for t in range(len(_types)) if _types[t] == 'RNN'] # we know exactly which layer(s) has RNN(s)
  m_h = [np.zeros_like(model[i][6]) for i in range(len(_indexes))] # not sure if this in _indexes is gonna work out properly, 
  v_h = [np.zeros_like(model[i][6]) for i in range(len(_indexes))]
  # TODO: check for more than 1 layers of RNN

  for i in range(len(model)):
    t+=1
    m_w[i] = b1 * m_w[i] + (1 - b1) * gradients[i][0] #estimates 1st momentum (mean) of gradient 
    v_w[i] = b2 * v_w[i] + (1 - b2) * np.square(gradients[i][0]) #estimates 2nd momentum (variance) of gradient 
    m_b[i] = b1 * m_b[i] + (1 - b1) * gradients[i][1]
    v_b[i] = b2 * v_b[i] + (1 - b2) * np.square(gradients[i][1])
    mhat_w = m_w[i] / (1. - b1**t) #corrects bias towards zero (initially set to vector of 0s, creates bias around it)
    vhat_w = v_w[i] / (1. - b2**t)
    mhat_b = m_b[i] / (1. - b1**t)
    vhat_b = v_b[i] / (1. - b2**t)
    model[i][0] -= lr*mhat_w / (np.sqrt(vhat_w) + eps) #updating weights
    model[i][1] -= lr*mhat_b / (np.sqrt(vhat_b) + eps) #updating biases

    if model[i][5] == Types.RNN:
      m_h[j] = b1 * m_h[j] + (1 - b1) * gradients[i][2] # here we don't need offset because we already know it is a RNN, so we know the exact index 
      v_h[j] = b2 * v_h[j] + (1 - b2) * np.square(gradients[i][2]) 
      mhat_h = m_h[j] / (1. - b1**t) 
      vhat_h = v_h[j] / (1. - b2**t)

      model[i][6] -= lr*mhat_h / (np.sqrt(vhat_h) + eps) # updating weights of hidden layer
      j += 1 # this is if we have more than one RNN layer

  return model


def RNN_AdamGrad(model, gradients, lr=0.001, b1=0.9, eps=1e-7):
  _types = [model[j][5].name for j in range(len(model))]
  # Adam doesn't have momentum (it estimates momentum), so we don't have to worry about that case

  t, j = 0, 0
  m_w = [np.zeros_like(model[i][0]) for i in range(len(model))]
  m_b = [np.zeros_like(model[i][1]) for i in range(len(model))]

  # Special case for RNN
  _indexes = [t for t in range(len(_types)) if _types[t] == 'RNN'] # we know exactly which layer(s) has RNN(s)
  m_h = [np.zeros_like(model[i][6]) for i in range(len(_indexes))] # not sure if this in _indexes is gonna work out properly, 
  # TODO: check for more than 1 layers of RNN

  t,j = 0,0

  for i in range(len(model)):
    m_w[i] = np.square(gradients[i][0])
    m_b[i] = np.square(gradients[i][1])
    model[i][0] -= lr*gradients[i][0] / (np.sqrt(m_w[i]) + eps) 
    model[i][1] -= lr*gradients[i][1] / (np.sqrt(m_b[i]) + eps) 

    if model[i][5] == Types.RNN:
      m_h[j] = np.square(gradients[i][2])

      model[i][6] -= lr*gradients[i][2]/ (np.sqrt(m_h[j]) + eps) 
      j += 1 

  return model    


# Adam (Adaptive Moment Estimation): the greates of them all.
# A simple combination of RMSProp + Moentum approach
def Adam(model, gradients, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
  _types = [model[j][5] for j in range(len(model))]
  # Adam doesn't have momentum (it estimates momentum), so we don't have to worry about that case
  if np.any(_types == Types.RNN):
    offset  = 0 
  else:
    offset  = 1

  t, j = 0, 0
  m_w = [np.zeros_like(model[i][0]) for i in range(len(model))]
  v_w = [np.zeros_like(model[i][0]) for i in range(len(model))]
  m_b = [np.zeros_like(model[i][1]) for i in range(len(model))]
  v_b = [np.zeros_like(model[i][1]) for i in range(len(model))]

  # Special case for RNN
  _indexes = np.argwhere(_types == Types.RNN) # we know exactly which layer(s) has RNN(s)
  m_h = [np.zeros_like(model[i][6]) for i in _indexes] # not sure if this in _indexes is gonna work out properly, TODO: check for more than 1 layers of RNN
  v_h = [np.zeros_like(model[i][6]) for i in _indexes]

  for i in range(len(model)):
    t+=1
    m_w[i] = b1 * m_w[i] + (1 - b1) * gradients[-1-i][0+offset] #estimates 1st momentum (mean) of gradient 
    v_w[i] = b2 * v_w[i] + (1 - b2) * np.square(gradients[-1-i][0+offset]) #estimates 2nd momentum (variance) of gradient 
    m_b[i] = b1 * m_b[i] + (1 - b1) * gradients[-1-i][1+offset]
    v_b[i] = b2 * v_b[i] + (1 - b2) * np.square(gradients[-1-i][1+offset])
    mhat_w = m_w[i] / (1. - b1**t) #corrects bias towards zero (initially set to vector of 0s, creates bias around it)
    vhat_w = v_w[i] / (1. - b2**t)
    mhat_b = m_b[i] / (1. - b1**t)
    vhat_b = v_b[i] / (1. - b2**t)
    model[i][0] -= lr*mhat_w / (np.sqrt(vhat_w) + eps) #updating weights
    model[i][1] -= lr*mhat_b / (np.sqrt(vhat_b) + eps) #updating biases

    if model[i][5] == Types.RNN:
      m_h[j] = b1 * m_h[j] + (1 - b1) * gradients[-1-i][2] # here we don't need offset because we already know it is a RNN, so we know the exact index 
      v_h[j] = b2 * v_h[j] + (1 - b2) * np.square(gradients[-1-i][2]) 
      mhat_h = m_h[j] / (1. - b1**t) 
      vhat_h = v_h[j] / (1. - b2**t)

      model[i][6] -= lr*mhat_h / (np.sqrt(vhat_h) + eps) # updating weights of hidden layer
      j += 1 # this is if we have more than one RNN layer

  return model

# RMSProp and Momentum take contrasting approaches. 
# While momentum accelerates our search in direction of minima, 
# RMSProp impedes our search in direction of oscillations.
# That's why RMSProp doesn't have momentum
def RMSProp(model, gradients, lr=0.001, b1=0.9, eps=1e-7):
  _types = [model[j][5] for j in range(len(model))]
  # Adam doesn't have momentum (it estimates momentum), so we don't have to worry about that case
  if np.any(_types == Types.RNN):
    offset  = 0 
  else:
    offset  = 1

  m_w = [np.zeros_like(model[i][0]) for i in range(len(model))]
  m_b = [np.zeros_like(model[i][1]) for i in range(len(model))]

  # Special case for RNN
  _indexes = np.argwhere(_types == Types.RNN) 
  m_h = [np.zeros_like(model[i][6]) for i in _indexes] 

  j = 0

  for i in range(len(model)):
    m_w[i] = b1 * m_w[i] + (1 - b1) * np.square(gradients[-1-i][0+offset])
    m_b[i] = b1 * m_b[i] + (1 - b1) * np.square(gradients[-1-i][1+offset])
    model[i][0] -= lr*gradients[-1-i][0+offset] / (np.sqrt(m_w[i]) + eps) #updating weights
    model[i][1] -= lr*gradients[-1-i][1+offset] / (np.sqrt(m_b[i]) + eps) #updating biases

    if model[i][5] == Types.RNN:
      m_h[j] = b1 * m_h[j] + (1 - b1) * np.square(gradients[-1-i][2])

      model[i][6] -= lr*gradients[-1-i][2]/ (np.sqrt(m_h[j]) + eps) 
      j += 1 

  return model    
