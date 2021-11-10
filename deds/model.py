import numpy as np 
import time
from tqdm import tqdm
from enum import Enum, unique
from deds import activation
from deds import losses
from deds import optimizers

# **** Model is The Constructor Basis of the Neural Network ***
# model is a list where:
# len(model) = number of layers (counting the input and output)
# model[i][0] is the Weights matrix of layer i
# model[i][1] is the bias vector of layer i
# model[i][2] is the activation of the layer i <<-- limitation: currently we can only have the same activation for the entire layer
# model[i][3] is the regularization of the layer i
# model[i][4] is the factor of the regularization (float)
# model[i][5] is the Type of the Layer (currently availables: Input, Output, Linear, RNN)

# If the layer is a RNN, there will be a model[i][6] which is the Hidden State Weights Matrix and a model[i][7] for the number of timesteps

# If the user adds dropout to the layer, there will be another entry where:
#   model[i][0] == 'Dropout'
#   model[i][1] == dropout factor (float)

@unique
class Types(Enum):
  Input = 'Input'
  Output = 'Output'
  Linear = 'Linear'
  RNN = 'RNN'
  Dropout = 'Dropout'

@unique
class Regs(Enum):
  L1 = 'L1'
  L2 = 'L2'
  No = None

class Dense:
  def __init__(self):
    pass

  def forward(self, layer, x):
    # W -- weight nxm matrix of for the next layer
    # x -- input vector of shape (m,1)
    # b -- bias vector of shape (n,1)
    # n -- number of output neurons (neurons of next layer)
    # m -- number of input neurons (neurons in the previous layer)
    # z[i] = W[i-1]x + b[i-1]
    # a[i] = f(z[i]) -- f is the activation funciton

    # TODO: Incorporate Dropout
    _type = layer[5]

    if _type in (Types.Input, Types.Linear, Types.Output):
      W = layer[0]
      b = layer[1]
      actv = layer[2]
      act = getattr(activation, actv)
      res = np.dot(W,x) + b
      return [x, res, act(res)]

    elif _type == Types.Dropout: 
      W = layer[0]
      b = layer[1]
      actv = layer[2]
      act = getattr(activation, actv)
      p = layer[4]
      y = np.random.binomial(1, p, size=(x.shape[0],1)).astype('float64')
      x *= y
      res = np.dot(W,x) + b
      return [x, res, act(res), p]

    else:
      raise ValueError(f' {_type} is an Incorrect type of Layer')


  def backward(self, A, actv, y, loss, _type, next_model, all_loss):
    # we do backpropagation
    # Output layer:
    # dc_dw_o = (dc_da_o)*(da_dz_o)*(dz_dw_o)
    # dc_db_o = (dc_da_o)*(da_dz_o)(dz_db_o) -- (dz_db_o) == 1 because z = wx+b
    # Hidden/Input layer(s):
    # dc_dw = (dc_da)*(da_dz)*(dz_dw)
    # but dc_da = (dc_da_t1)*(da_t1_dz_t1)*(dz_t1_da), where t1 means one layer above
    # dc_db_ = (dc_da)*(da_dz)(dz_db) -- (dz_db) == 1 because z = wx+b

    # TODO: Dropout is this segment comment below!
    W_t1 = next_model[0] # if A[-1] == None else next_model[0]*A[-1]
    a_t0 = A[0]
    z = A[1]
    a = A[2]
    d_loss_ = getattr(losses, 'd'+loss)
    d_act_ = getattr(activation, 'd'+actv)

    if _type == Types.Output:
      all_lost = list()
      #dc_dw
      dc_dz_o = (a - y)
      dz_dw_o = a_t0.T #previous activation
      dc_dw_o = np.dot(dc_dz_o, dz_dw_o)/y.shape[0]

      #dc_db
      dc_db_o = dc_dz_o/y.shape[0]
      return [dc_dz_o, dc_dw_o, dc_db_o]

    else:
      #dc_dw
      dc_dz_t1 = all_loss[0]
      dz_t1_da = W_t1
      da_dz = d_act_(a)
      dc_dz = np.dot((dc_dz_t1).T, dz_t1_da).T *da_dz
      dz_dw = a_t0.T
      dc_dw = np.dot(dc_dz, dz_dw)/y.shape[0]

      #dc_db
      dc_db = dc_dz/y.shape[0]
      return [dc_dz, dc_dw,dc_db]


  def summary(self, model):
    print(f'| Total Number of Layers: {len(model)} |')
    for i in range(len(model)):
      inputs = model[i][0].shape[1]
      outputs = model[i][0].shape[0]
      _type = model[i][5].name 

      print(f'| layer {i+1} of type {_type} with {inputs} inputs and {outputs} outputs neurons |')

  def Input(self, neurons, input_shape, activation, reg=None, reg_num=0):
    #random weights and bias between -0.5 to 0.5
    np.random.seed(23)
    weights = np.random.rand(neurons, input_shape) - 0.5
    bias = np.random.rand(neurons, 1) - 0.5
    return [[weights, bias, activation, Regs(reg), reg_num, Types('Input')]]

  def Output(self, pr_neurons, next_neurons, model, activation, reg=None, reg_num=0):
    np.random.seed(23)
    weights = np.random.rand(next_neurons, pr_neurons) - 0.5
    bias = np.random.rand(next_neurons, 1) - 0.5
    model.append([weights, bias, activation, Regs(reg), reg_num, Types('Output')])
    return model

  def Linear(self, pr_neurons, next_neurons, model, activation, reg=None, reg_num=0):
    np.random.seed(23)
    weights = np.random.rand(next_neurons, pr_neurons) - 0.5
    bias = np.random.rand(next_neurons, 1) - 0.5
    model.append([weights, bias, activation, reg, reg_num, Types('Linear')])
    return model

  # TODO: Refactor Dropout. Simply adding to the last model entry is bad
  # because model[i][5] can be RNN or Dropout
  def Dropout(self, model, p):
    if p < 0 or p > 1:
      raise ValueError("Dropout Probability has to be between 0 and 1, but got {}".format(p))
    model[-1].append(Types('Dropout'))
    model[-1].append(p)
    return model

  def Compile(self, optimizer, loss, metrics, lr= 0.001, momentum=True, gamma=0.95):
    self.optimizer = optimizer
    self.loss = loss
    self.lr = lr

    #optimizer with momentum
    if momentum:
      exceptions = ['Adam', 'RMSProp'] #optimizers that doesn't have momentum
      assert self.optimizer not in exceptions, f'{self.optimizer} optimizer does not have momentum property'
      self.momentum = True
      self.gamma = gamma
    else:
      self.momentum = False
      self.gamma = 0

  def Train(self, model, x, y, epochs, batch, categoric):
    l = []
    ac = []
    loss_ = getattr(losses, self.loss)
    opt_ = getattr(optimizers, self.optimizer)

    #print summary
    self.summary(model)

    pbar = tqdm(range(epochs))
    timing = epochs/(epochs*100) 
    for i,pb in enumerate(pbar):
      pbar.set_description(f"Training... epoch {i}/{epochs}")
      time.sleep(timing) 
      avg_loss = 0
      acc = 0
      count = 0
      samp = np.random.randint(0, y.shape[0], size=y.shape[0]//batch)
      #samp = np.arange(0, y.shape[0], batch) #batch sample size
      for k in samp:
        A = list()
        
        count += 1

        #forward pass
        for j in range(len(model)):
          if (j == 0):
            A.append(self.forward(model[j], x[k]))
          else:
            A.append(self.forward(model[j], A[-1][2]))

        #backward pass
        all_loss = list()
        for j in range(len(model)):
          if j == 0:
            all_loss.append(self.backward(A[-1-j], model[-1-j][2], y[k], self.loss, model[-1-j][5], model[-j], [0,0,0]))
          else:
            all_loss.append(self.backward(A[-1-j], model[-1-j][2], y[k], self.loss, model[-1-j][5], model[-j], all_loss[-1]))

        reg = 0
        #weight regularization
        reg_1 = [model[i][4]*abs(all_loss[i][1]).mean() for i in range(len(model)) if model[i][3] == Regs.L1]
        reg_1 = np.mean(reg_1) if len(reg_1) > 0 else 0  
        reg_2 = [model[i][4]*(all_loss[i][1]**2).mean() for i in range(len(model)) if model[i][3] == Regs.L2]
        reg_2 = np.mean(reg_2) if len(reg_2) > 0 else 0  
        reg = reg_1 + reg_2

        #loss
        avg_loss += (loss_(A[-1][2], y[k])).mean() + reg
        
        if categoric:
          if (np.argmax(A[-1][2]) == np.argmax(y[k])):
            acc += 1

        else:
          if (int(A[-1][2][0]) == int(y[k])):
            acc += 1

        if self.momentum:
          if count == 1:
            gradients = [[self.lr*all_loss[j][1], self.lr*all_loss[j][2]] for j in range(len(model))]
          else:
            gradients = [[self.gamma*gradients[j][0] + self.lr*all_loss[j][1],
                    self.gamma*gradients[j][1] + self.lr*all_loss[j][2]] for j in range(len(model))]
        else:
          gradients = all_loss
        #update params
        if self.optimizer == 'SGD':                 
          model = opt_(model, gradients, self.momentum, self.lr) 
        elif self.optimizer == 'RMSProp':
          model = opt_(model, gradients, self.lr)
        elif self.optimizer == 'Adam':
          model = opt_(model, gradients, self.lr) 
        else:
          model = opt_(model, gradients, self.momentum) 

      acc /= count
      avg_loss /= count

      print(f'accuracy: {acc}, loss: {avg_loss}')
      l.append(avg_loss)
      ac.append(acc)

    return model, l, ac

  def Evaluate(self, model, x, y, categoric):
    results = list()
    precision = 0
    for k in range(len(x)):
      for j in range(len(model)):
        if (j == 0):
          results.append(self.forward(model[j], x[k]))
        else:
          results.append(self.forward(model[j], results[-1][2]))

      if categoric:
        if (np.argmax(results[-1][2]) == np.argmax(y[k])):
          precision += 1

      else:
        if (int(results[-1][2][0]) == int(y[k])):
          precision += 1


    print(f'Network got {precision/len(y)} right')
    return precision

class RNN:
  def __init__(self):
    pass

  def forward(self, layer, x, time_steps):
    # W -- weight nxm matrix of for the next layer
    # x -- input vector of shape (m,1)
    # b -- bias vector of shape (n,1)
    # n -- number of output neurons (neurons of next layer)
    # m -- number of input neurons (neurons in the previous layer)
    # z[i] = W[i-1]x + b[i-1]
    # a[i] = f(z[i]) -- f is the activation funciton

    # TODO: Incorporate Dropout
    _type = layer[5]
    res = np.zeros((time_steps,))

    if _type in (Types.Input, Types.Linear, Types.Output, Types.RNN):
      for t in range(time_steps):
        W[t] = layer[0]
        b[t] = layer[1]
        actv = layer[2]
        act = getattr(activation, actv)
        res[t] = np.dot(W[t],x[t]) + b[t]
      return [x, res, act(res)]

    if _type == Types.RNN:
      res = np.zeros((time_steps,))
      for t in range(time_steps):
        W[t] = layer[0]
        b[t] = layer[1]
        actv = layer[2]
        act = getattr(activation, actv)
        res[t] = np.dot(W[t],x[t]) + np.dot(Wh[t], res[t-1]) + b[t]
      return [x, res, act(res)]

    else:
      raise ValueError(f' {_type} is an Incorrect type of Layer')


  def backward(self, A, model, y, loss, _type, next_model, all_loss, time_steps):

    # TODO: Implement Dropout! The comment part below is from that
    W_t1 = next_model[0] 
    a_t0 = A[0] # previous activation
    z = A[1]
    a = A[2]
    actv = model[2]
    d_loss_ = getattr(losses, 'd'+loss)
    d_act_ = getattr(activation, 'd'+actv)

    if _type == Types.Output:
      all_lost = list()

      for t in range(time_steps):
        # dc_dw
        dc_dz_o = (a[t] - y[t]) # TODO: mudar para d_loss 
        dc_dw_o += np.dot(dc_dz_o, a_t0[t].T)/len(y)

        # dc_db
        dc_db_o += dc_dz_o/len(y)

      return [dc_dz_o, dc_dw_o, dc_db_o]

    elif _type == Types.RNN:
      Whh = model[6] 
      dc_dwhh = np.zeros_like(Whh)

      for t in range(time_steps):
        # Compute hidden state
        hnext = np.zeros((Whh.shape[0], W_t1.shape))

        # dc_dw
        dc_da = np.dot(all_loss[0].T, W_t1).T
        da_dw = np.dot(d_act_(z[t]), a_t0[t].T) + d_act_(z[t])* np.dot(Whh, hnext)
        dc_dw += dc_da*da_dw

        # dc_db
        dc_db += dc_da * d_actz(z[t]) 
 
        # update param of hidden state
        hnext += da_dw

        # dc_dwhh
        dc_dz = np.dot(all_loss[0].T, W_t1).T * d_act_(z[t])
        dz_dw = a[t-1].T if t!=0 else np.zeros_like(a[0]).T
        dc_dwhh += np.dot(dc_dz, dz_dw)

      return [dc_dz, dc_dw,dc_db, dc_dwhh]


    else:
      for t in range(time_steps):
        # dc_dw
        dc_dz_t1 = all_loss[0]
        dc_dz = np.dot( all_loss[0].T, W_t1).T * d_act_(a[t])
        dc_dw += np.dot(dc_dz, a_t0[t].T)/len(y)

        # dc_db
        dc_db += dc_dz/len(y)

      return [dc_dz, dc_dw,dc_db]


  def summary(self, model):
    print(f'| Total Number of Layers: {len(model)} |')
    for i in range(len(model)):
      inputs = model[i][0].shape[1]
      outputs = model[i][0].shape[0]
      _type = model[i][5].name 

      print(f'| layer {i+1} of type {_type} with {inputs} inputs and {outputs} outputs neurons |')

  def Input(self, neurons, input_shape, activation, reg=None, reg_num=0):
    #random weights and bias between -0.5 to 0.5
    np.random.seed(23)
    weights = np.random.rand(neurons, input_shape) - 0.5
    bias = np.random.rand(neurons, 1) - 0.5
    return [[weights, bias, activation, Regs(reg), reg_num, Types('Input')]]

  def Output(self, pr_neurons, next_neurons, model, activation, reg=None, reg_num=0):
    np.random.seed(23)
    weights = np.random.rand(next_neurons, pr_neurons) - 0.5
    bias = np.random.rand(next_neurons, 1) - 0.5
    model.append([weights, bias, activation, Regs(reg), reg_num, Types('Output')])
    return model

  def Linear(self, pr_neurons, next_neurons, model, activation, reg=None, reg_num=0):
    np.random.seed(23)
    weights = np.random.rand(next_neurons, pr_neurons) - 0.5
    bias = np.random.rand(next_neurons, 1) - 0.5
    model.append([weights, bias, activation, reg, reg_num, Types('Linear')])
    return model

  def RNN(self, pr_neurons, next_neurons, hidden_neurons, model, timesteps, reg=None, reg_num=0):
    np.random.seed(23)
    weights = np.random.rand(next_neurons, pr_neurons) - 0.5
    hidden_state = np.random.rand(hidden_neurons, hidden_neurons) - 0.5
    bias = np.random.rand(next_neurons, 1) - 0.5
    model.append([weights, bias, 'tanh', reg, reg_num, Types('RNN'), timesteps])
    return model

  def Compile(self, optimizer, loss, metrics, lr= 0.001, momentum=True, gamma=0.95):
    self.optimizer = optimizer
    self.loss = loss
    self.lr = lr

    #optimizer with momentum
    if momentum:
      exceptions = ['Adam', 'RMSProp'] #optimizers that doesn't have momentum
      assert self.optimizer not in exceptions, f'{self.optimizer} optimizer does not have momentum property'
      self.momentum = True
      self.gamma = gamma
    else:
      self.momentum = False
      self.gamma = 0

  def Train(self, model, x, y, epochs, batch, categoric):
    l = []
    ac = []
    loss_ = getattr(losses, self.loss)
    opt_ = getattr(optimizers, self.optimizer)

    #print summary
    self.summary(model)

    pbar = tqdm(range(epochs))
    timing = epochs/(epochs*100) 
    for i,pb in enumerate(pbar):
      pbar.set_description(f"Training... epoch {i}/{epochs}")
      time.sleep(timing) 
      avg_loss = 0
      acc = 0
      count = 0
      samp = np.random.randint(0, y.shape[0], size=y.shape[0]//batch)
      #samp = np.arange(0, y.shape[0], batch) #batch sample size
      for k in samp:
        A = list()
        
        count += 1

        #forward pass
        for j in range(len(model)):
          if (j == 0):
            A.append(self.forward(model[j], x[k]))
          else:
            A.append(self.forward(model[j], A[-1][2]))

        #backward pass
        all_loss = list()
        for j in range(len(model)):
          if j == 0:
            all_loss.append(self.backward(A[-1-j], model[-1-j][2], y[k], self.loss, model[-1-j][5], model[-j], [0,0,0]))
          else:
            all_loss.append(self.backward(A[-1-j], model[-1-j][2], y[k], self.loss, model[-1-j][5], model[-j], all_loss[-1]))

        reg = 0
        #weight regularization
        reg_1 = [model[i][4]*abs(all_loss[i][1]).mean() for i in range(len(model)) if model[i][3] == Regs.L1]
        reg_1 = np.mean(reg_1) if len(reg_1) > 0 else 0  
        reg_2 = [model[i][4]*(all_loss[i][1]**2).mean() for i in range(len(model)) if model[i][3] == Regs.L2]
        reg_2 = np.mean(reg_2) if len(reg_2) > 0 else 0  
        reg = reg_1 + reg_2

        #loss
        avg_loss += (loss_(A[-1][2], y[k])).mean() + reg
        
        if categoric:
          if (np.argmax(A[-1][2]) == np.argmax(y[k])):
            acc += 1

        else:
          if (int(A[-1][2][0]) == int(y[k])):
            acc += 1

        if self.momentum:
          if count == 1:
            gradients = [[self.lr*all_loss[j][1], self.lr*all_loss[j][2]] for j in range(len(model))]
          else:
            gradients = [[self.gamma*gradients[j][0] + self.lr*all_loss[j][1],
                    self.gamma*gradients[j][1] + self.lr*all_loss[j][2]] for j in range(len(model))]
        else:
          gradients = all_loss
        #update params
        if self.optimizer == 'SGD':                 
          model = opt_(model, gradients, self.momentum, self.lr) 
        elif self.optimizer == 'RMSProp':
          model = opt_(model, gradients, self.lr)
        elif self.optimizer == 'Adam':
          model = opt_(model, gradients, self.lr) 
        else:
          model = opt_(model, gradients, self.momentum) 

      acc /= count
      avg_loss /= count

      print(f'accuracy: {acc}, loss: {avg_loss}')
      l.append(avg_loss)
      ac.append(acc)

    return model, l, ac


