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


  def backward(self, A, actv, y, loss, _type, next_model, back):
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
      #dc_dw
      dc_dz_o = (a - y)
      dz_dw_o = a_t0.T #previous activation
      dc_dw_o = np.dot(dc_dz_o, dz_dw_o)/y.shape[0]

      #dc_db
      dc_db_o = dc_dz_o/y.shape[0]
      return [dc_dz_o, dc_dw_o, dc_db_o]

    else:
      #dc_dw
      dc_dz_t1 = back[0]
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
        back = list()
        for j in range(len(model)):
          if j == 0:
            back.append(self.backward(A[-1-j], model[-1-j][2], y[k], self.loss, model[-1-j][5], model[-j], [0,0,0]))
          else:
            back.append(self.backward(A[-1-j], model[-1-j][2], y[k], self.loss, model[-1-j][5], model[-j], back[-1]))

        reg = 0
        #weight regularization
        reg_1 = [model[i][4]*abs(back[i][1]).mean() for i in range(len(model)) if model[i][3] == Regs.L1]
        reg_1 = np.mean(reg_1) if len(reg_1) > 0 else 0  
        reg_2 = [model[i][4]*(back[i][1]**2).mean() for i in range(len(model)) if model[i][3] == Regs.L2]
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
            gradients = [[self.lr*back[j][1], self.lr*back[j][2]] for j in range(len(model))]
          else:
            gradients = [[self.gamma*gradients[j][0] + self.lr*back[j][1],
                    self.gamma*gradients[j][1] + self.lr*back[j][2]] for j in range(len(model))]
        else:
          gradients =back 
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

  def forward(self, model, x, y, inputs, outputs, m):
    A = list()
    self.avg_loss = 0
    hidden = np.zeros((m, self.vocab_size))

    for t in range(m):
      x[t][inputs] = 1
      y[t][outputs] = 1


      for j in range(len(model)):
        _type = model[j][5]
        act = getattr(activation, model[j][2])

        assert _type in (Types.Input, Types.Linear, Types.Output, Types.RNN), f'Incorrect type. Got {_type}, but we only support Input, Linear, RNN, Output'

        if j == 0:
          z = np.dot(model[j][0],x[t].reshape(-1,1)) + model[j][1]

          A.append([x[t].reshape(-1,1), z, act(z)])
       
        else:
          
          if _type == Types.RNN:
            assert act == activation.Tanh

            hidden += np.dot(model[j][6], hidden)
            z = np.dot(model[j][0],A[-1][2]) + hidden + model[j][1] 
            A.append([A[-1][2], z, act(z)])

          else:
            z = np.dot(model[j][0],A[-1][2]) + model[j][1] 
            A.append([A[-1][2], z, act(z)])


      # compute loss
      self.avg_loss += -(np.log(A[-1][2][outputs[t],0])) #a bit weird loss

    return A


  def backward(self, A, model, y, m):

    back = list()
    gradients = list()
    dz_dw = np.zeros_like(A[0][2][0]).T # TODO: don't know if this is correct

    for j in range(len(model)):
      if model[j][5] == Types.RNN:
        # weights of layer, bias , weight of cell
        gradients.append([np.zeros_like(model[j][0]), np.zeros_like(model[j][1]) , np.zeros_like(model[j][6])])
      else:
        # weight of layer, bias
        gradients.append([np.zeros_like(model[j][0]), np.zeros_like(model[j][1])])

    i = len(A) - 1
    for t in reversed(range(m)):

      for j in reversed(range(len(model))):
        d_loss_ = getattr(losses, 'd'+self.loss)
        d_act_ = getattr(activation, 'd'+model[j][2])
        _type = model[j][5]
        a_t0 = A[i][0] # previous activation
        z = A[i][1]
        a = A[i][2]


        if _type == Types.Output:

          # dc_dw
          dc_dz_o = (A[i][2] - y[t].reshape(-1,1)) # TODO: mudar para d_loss 
          gradients[j][0] += np.dot(dc_dz_o, A[i][0].T)/len(y[t])

          # dc_db
          gradients[j][1] += dc_dz_o/len(y[t])


        elif _type == Types.RNN:
          W_t1 = model[j+1][0] # weight of next layer
          Whh = model[j][6] 
          hnext = np.zeros((Whh.shape[0], W_t1.shape[0])) # TODO: maybe one for each RNN layer?

          # dc_dw
          dc_da = np.dot(dc_dz_o.T, model[j+1][0]).T
          da_dw = np.dot(d_act_(A[i][1]), A[i][0].T) + d_act_(A[i][1])* np.dot(Whh, hnext)
          gradients[j][0] += dc_da*da_dw

          # dc_db
          gradients[j][1] += dc_da * d_act_(A[i][1]) 
   
          # update param of hidden state
          hnext += da_dw

          # dc_dwhh
          dc_dz = np.dot(dc_dz_o.T, model[j+1][0]).T * d_act_(A[i][1])
          dz_dw = A[i][2][t-1].T # TODO: check this 
          gradients[j][2] += np.dot(dc_dz, dz_dw)


        else:
          W_t1 = model[j+1][0] # weight of next layer

          # dc_dw
          dc_dz_t1 = dc_dz 
          dc_dz = np.dot(dc_dz.T, W_t1).T * d_act_(A[i][2])
          gradients[j][0] += np.dot(dc_dz, A[i][0].T)/len(y)

          # dc_db
          gradients[j][1] += dc_dz/len(y)

        i -= 1

    return gradients

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
    weights = np.random.randn(neurons, input_shape) * 0.01
    bias = np.random.randn(neurons,1) * 0.01
    return [[weights, bias, activation, Regs(reg), reg_num, Types('Input')]]

  def Output(self, pr_neurons, next_neurons, model, activation, reg=None, reg_num=0):
    np.random.seed(23)
    weights = np.random.randn(next_neurons, pr_neurons) * 0.01 
    bias = np.random.randn(next_neurons,1) * 0.01 
    model.append([weights, bias, activation, Regs(reg), reg_num, Types('Output')])
    return model

  def Linear(self, pr_neurons, next_neurons, model, activation, reg=None, reg_num=0):
    np.random.seed(23)
    weights = np.random.randn(next_neurons, pr_neurons) * 0.01 
    bias = np.random.randn(next_neurons,1) * 0.01 
    model.append([weights, bias, activation, reg, reg_num, Types('Linear')])
    return model

  def RNN(self, pr_neurons, next_neurons, hidden_neurons, model, seq_length, reg=None, reg_num=0):
    np.random.seed(23)
    self.hidden_size = hidden_neurons
    self.vocab_size = pr_neurons
    weights = np.random.randn(next_neurons, pr_neurons) * 0.01 
    hidden_state = np.random.randn(hidden_neurons, hidden_neurons) * 0.01 
    bias = np.random.randn(next_neurons,1) * 0.01 
    self.seq_length = seq_length

    if model == None:
      model = [([weights, bias, 'Tanh', reg, reg_num, Types('RNN'), hidden_state, seq_length])]
      return model 
    else:
      model.append([weights, bias, 'Tanh', reg, reg_num, Types('RNN'), hidden_state, seq_length])
      return model

  def Compile(self, optimizer, loss, metrics, time_step, lr= 0.001, momentum=True, gamma=0.95):
    self.optimizer = optimizer
    self.loss = loss
    self.lr = lr
    self.time_step = time_step

    #optimizer with momentum
    if momentum:
      exceptions = ['Adam', 'RMSProp'] #optimizers that doesn't have momentum
      assert self.optimizer not in exceptions, f'{self.optimizer} optimizer does not have momentum property'
      self.momentum = True
      self.gamma = gamma
    else:
      self.momentum = False
      self.gamma = 0

  def Train(self, model, data, char_to_ix, ix_to_char, epochs=100000, batch=32):

    #Training (trying to predict next character)
    n,p = 0,0

    #ADAMGRAD
    mWxh = np.zeros((self.hidden_size, self.vocab_size))
    mWhh = np.zeros((self.hidden_size, self.hidden_size))
    mWhy = np.zeros((self.vocab_size, 1))
    mbh = np.zeros((self.hidden_size, 1))
    mby = np.zeros((self.vocab_size,1)) 

    hprev = np.zeros((self.hidden_size, 1))

    while n<= epochs:
        #when we finish the training data
        if p+self.seq_length+1 >= len(data) or n == 0:
            hprev = np.zeros((self.hidden_size, 1)) #FOR FORWARD PASS AT SAMPLE FUNCTION!!
            p = 0 #start again the training data
        
        #vetores de indices
        inputs = [char_to_ix[ch] for ch in data[p:p+self.seq_length]]
        outputs = [char_to_ix[ch] for ch in data[p+1:p+self.seq_length+1]]

        #initializing vectors
        m = len(inputs)
        x, y = np.zeros((m, self.vocab_size,)), np.zeros((m, self.vocab_size,))
        
        #forward
        A = self.forward(model, x, y, inputs, outputs, m)
        print('Success Forward!')

        #gradients
        #dWxh = np.zeros_like(Wxh)
        #dbh = np.zeros_like(bh)
        #dWhh = np.zeros_like(Whh)
        #dWhy = np.zeros_like(Why)
        #dby = np.zeros_like(by)
        #hnext = np.zeros((self.hidden_size, self.vocab_size))

        # backward
        back = self.backward(A, model, y, m)
        print('Success Backward!')

        print(f'epoch: {n}, loss: {self.avg_loss}') 
        return

        # att hprev
        hprev = a_1[-1]

        #sample from model to see the performance (just for showing off)
        if n % 1000 == 0:
            print(f'epoch: {n}, loss: {self.avg_loss}') 

            #generate 200 characters to see how the network is
            x = np.zeros((self.vocab_size,1))
            x[inputs[0]] = 1
            indexes = []
            h = hprev
            for t in range(200):
                h = activation.Tanh(np.dot(Wxh, x) + np.dot(Whh, h))# + bh)
                predict = softmax(np.dot(Why, h))# + by)

                ix = np.random.choice(range(self.vocab_size), p=predict.ravel())

                #saving the predicted character 
                x = np.zeros((self.vocab_size,1))
                x[ix] = 1

                indexes.append(ix)

            txt = ''.join(ix_to_char[ix] for ix in indexes)
            print('----\n {} \n----'.format(txt))



        #optimizing parameters
        for param, dparam, mem in zip([Wxh, Whh, Why],
            [dWxh, dWhh, dWhy],
            [mWxh, mWhh, mWhy]):

            mem += dparam * dparam
            param += -lr * dparam / np.sqrt(mem + 1e-8) # adagrad update


        p += self.seq_length # move data pointer
        n += 1      

