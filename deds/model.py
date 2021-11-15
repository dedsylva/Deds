import numpy as np 
import time
from tqdm import tqdm
from deds import activation
from deds import losses
from deds import optimizers
from deds.utils.enumerators import Types, Regs

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
      samp = np.random.randint(0, y.shape[0], size=y.shape[0]//batch) #batches
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
          gradients = back 
        #update params
        if self.optimizer == 'SGD':                 
          model = opt_(model, gradients, self.momentum, self.lr) 
        elif self.optimizer == 'RMSProp':
          model = opt_(model, gradients, self.lr)
        elif self.optimizer == 'Adam':
          model = opt_(model, gradients, self.lr) 
        else:
          raise ValueError(f'Optimizer not supported. Currently available ones: SGD, RMSProp, Adam, but got {self.optimizer} instead') 

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
    loss_ = getattr(losses, self.loss)

    # store the hidden_states of every RNN layer
    _types = [model[j][5].name for j in range(len(model))]
    _indexes = [t for t in range(len(_types)) if _types[t] == 'RNN'] # we know exactly which layer(s) has RNN(s)
    #hidden = [0 for i in range(len(_indexes))]
    #hidden[0] = 3
    #print(hidden, hidden[0])

    h_t0 = np.zeros((self.hidden_size,1)) # hidden state of time = 0
    for t in range(m):
      x[t][inputs[t]] = 1
      y[t][outputs[t]] = 1


      for j in range(len(model)):
        _type = model[j][5]
        act = getattr(activation, model[j][2])
        _input = x[t].reshape(-1,1) if j == 0 else A[-1][2]

        assert _type in (Types.Input, Types.Linear, Types.Output, Types.RNN), f'Incorrect type. Got {_type}, but we only support Input, Linear, RNN, Output'

        if _type == Types.RNN:
          assert act == activation.Tanh, f'Wrong activation layer. RNN uses Tanh, but got {act}'

          h = np.dot(model[j][0],_input) + np.dot(model[j][6], h_t0) + model[j][1] 
          A.append([_input, h, act(h)])

          # hprev
          h_t0 = act(h)


        else:
          z = np.dot(model[j][0], _input) + model[j][1] 
          A.append([_input, z, act(z)])

      # compute loss
      self.avg_loss += -(np.log(A[-1][2][outputs[t],0])) # TODO: change this to loss_()
      possible_loss = abs(np.log(A[-1][2][outputs[t],0])) 
      if possible_loss < 1e-4: 
        print(f'Close to zero in computing log, got {possible_loss}')

    #print(A[0][2])
    #print(A[-1][2])
    #exit(0)

    return h_t0, A


  def backward(self, A, model, y, inputs, outputs, m):

    gradients = list()
    dz_dw_h = np.zeros_like(A[0][2][0]).T # TODO: don't know if this is correct

    for j in range(len(model)):
      if model[j][5] == Types.RNN:
        # weights of layer, bias , weight of cell, hidden time update
        gradients.append([np.zeros_like(model[j][0]), np.zeros_like(model[j][1]) , np.zeros_like(model[j][6]), np.zeros((model[j][6].shape[0], model[j+1][0].shape[0]))])
      else:
        # weight of layer, bias
        gradients.append([np.zeros_like(model[j][0]), np.zeros_like(model[j][1])])

    i = len(A) - 1
    for t in reversed(range(m)):
      #hnext = np.zeros((Whh.shape[0], W_t1.shape[0])) # TODO: maybe one for each RNN layer?
      y[t][outputs[t]] = 1

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

          dc_dz_t1 = dc_dz_o

        elif _type == Types.RNN:
          W_t1 = model[j+1][0] # weight of next layer
          Whh = model[j][6] 

          # dc_dw
          dc_da = np.dot(dc_dz_t1.T, model[j+1][0]).T
          da_dw = np.dot(d_act_(A[i][1]), A[i][0].T) + d_act_(A[i][1])* np.dot(Whh, gradients[j][3])
          gradients[j][0] += dc_da*da_dw

          # dc_db
          gradients[j][1] += dc_da * d_act_(A[i][1]) 
   
          # update param of hidden state
          gradients[j][3] += da_dw

          # dc_dwhh
          dc_dz_h = np.dot(dc_dz_t1.T, model[j+1][0]).T * d_act_(A[i][1])
          dz_dw_h = A[i][2][t-1].T 
          gradients[j][2] += np.dot(dc_dz_h, dz_dw_h)

          dc_dz_t1 = dc_da # TODO: check this

        else:
          W_t1 = model[j+1][0] # weight of next layer

          # dc_dw
          dc_dz = np.dot(dc_dz_t1.T, model[j+1][0]).T * d_act_(A[i][2])
          gradients[j][0] += np.dot(dc_dz, A[i][0].T)/len(y[t])

          # dc_db
          gradients[j][1] += dc_dz/len(y[t])

          dc_dz_t1 = dc_dz

        i -= 1

    return gradients

  def summary(self, model):
    print(f'| Total Number of Layers: {len(model)} |')
    for i in range(len(model)):
      _inputs = model[i][0].shape[1]
      _outputs = model[i][0].shape[0]
      _type = model[i][5].name 

      print(f'| layer {i+1} of type {_type} with {_inputs} inputs and {_outputs} outputs neurons |')

  def Input(self, neurons, input_shape, activation, reg=None, reg_num=0):
    #random weights and bias between -0.5 to 0.5
    np.random.seed(23)
    weights = np.random.randn(neurons, input_shape) * 0.01
    bias = np.zeros((neurons,1))
    return [[weights, bias, activation, Regs(reg), reg_num, Types('Input')]]

  def Output(self, pr_neurons, next_neurons, model, activation, reg=None, reg_num=0):
    np.random.seed(23)
    weights = np.random.randn(next_neurons, pr_neurons) * 0.01 
    bias = np.zeros((next_neurons,1))
    model.append([weights, bias, activation, Regs(reg), reg_num, Types('Output')])
    return model

  def Linear(self, pr_neurons, next_neurons, model, activation, reg=None, reg_num=0):
    np.random.seed(23)
    weights = np.random.randn(next_neurons, pr_neurons) * 0.01 
    bias = np.zeros((next_neurons,1))
    model.append([weights, bias, activation, reg, reg_num, Types('Linear')])
    return model

  def RNN(self, pr_neurons, next_neurons, hidden_neurons, model, seq_length, reg=None, reg_num=0):
    np.random.seed(23)
    self.hidden_size = hidden_neurons
    self.vocab_size = pr_neurons
    weights = np.random.randn(next_neurons, pr_neurons) * 0.01 
    hidden_state = np.random.randn(hidden_neurons, hidden_neurons) * 0.01 
    bias = np.zeros((next_neurons,1))
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
    opt_ = getattr(optimizers, 'RNN_'+self.optimizer)

    # the last hidden state, for creating samples from the model (should be optional)
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
        hprev, A = self.forward(model, x, y, inputs, outputs, m)

        # gradients for optimization
        #mWxh = np.zeros_like(model[0][0])
        #mbx = np.zeros_like(model[0][1])
        #mWhh = np.zeros_like(model[0][6])
        #mWhy = np.zeros_like(model[1][0])
        #mby = np.zeros_like(model[1][1])

        # backward
        gradients = self.backward(A, model, y, inputs, outputs, m)

        #optimizing parameters
        #mWxh += np.square(gradients[0][0])
        #mbx += np.square(gradients[0][1])
        #mWhh += np.square(gradients[0][2])
        #mWhy += np.square(gradients[1][0])
        #mby += np.square(gradients[1][1])

        for g in [gradients[0][0], gradients[0][1], gradients[0][2], gradients[1][0], gradients[1][1]]:
          np.clip(g, -5, 5, out=g)


        #sample from model to see the performance (just for showing off)
        if n % 1000 == 0:
          print(f'epoch: {n}, loss: {self.avg_loss}') 

          #generate 200 characters to see how the network is
          x = np.zeros((self.vocab_size,1))
          x[inputs[0]] = 1
          indexes = []
          h = hprev

          for t in range(200):

            h = activation.Tanh(np.dot(model[0][0], x) + np.dot(model[0][6], h) + model[0][1])
            predict = activation.Softmax(np.dot(model[1][0], h) + model[1][1])

            ix = np.random.choice(range(self.vocab_size), p=predict.ravel())

            #saving the predicted character 
            x = np.zeros((self.vocab_size,1))
            x[ix] = 1

            indexes.append(ix)

          txt = ''.join(ix_to_char[ix] for ix in indexes)
          print('----\n {} \n----'.format(txt))

        #update params
        if self.optimizer == 'SGD':                 
          model = opt_(model, gradients, self.momentum, self.lr) 
        elif self.optimizer == 'RMSProp':
          model = opt_(model, gradients, self.lr)
        elif self.optimizer == 'Adam':
          model = opt_(model, gradients, self.lr) 
        else:
          raise ValueError(f'Optimizer not supported. Currently available ones: SGD, RMSProp, Adam, but got {self.optimizer} instead') 



        #optimizing parameters
        #mWxh += np.square(gradients[0][0])
        #mbx += np.square(gradients[0][1])
        #mWhh += np.square(gradients[0][2])
        #mWhy += np.square(gradients[1][0])
        #mby += np.square(gradients[1][1])

        #model[0][0] -= self.lr * gradients[0][0] / np.sqrt(mWxh + 1e-8)
        #model[0][1] -= self.lr * gradients[0][1] / np.sqrt(mbx + 1e-8)
        #model[0][6] -= self.lr * gradients[0][2] / np.sqrt(mWhh + 1e-8)
        #model[1][0] -= self.lr * gradients[1][0] / np.sqrt(mWhy + 1e-8)
        #model[1][1] -= self.lr * gradients[1][1] / np.sqrt(mby + 1e-8)


        #if n % 1000 == 0:
          #print(model[0][0], model[0][6], model[1][0])


        p += self.seq_length # move data pointer
        n += 1      

