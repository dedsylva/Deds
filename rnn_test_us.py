import numpy as np

def relu(x):
    return np.maximum(0,x)

def dReLu(x):
    data = np.array(x, copy=True)
    data[x<= 0] = 0
    return data

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)   

def dtanh(x):
    return 1 - np.square(np.tanh(x))

def softmax(x):
    max_ = np.max(x)
    return np.exp(x-max_)/sum(np.exp(x-max_))

#source = 'harry_potter.txt'
source = 'deds/datasets/kafka.txt'
data = open(source, 'r', encoding='UTF-8').read()
chars = list(set(data)) #set filters unique characters already
data_size, vocab_size = len(data), len(chars)

#hot encoding (sparse, but just a example, should use word embedding)
char_to_ix = { ch:i for i,ch in enumerate(chars)}
ix_to_char = { i:ch for i,ch in enumerate(chars)}

#example of hot encoding for a
vector_for_char_a = np.zeros((vocab_size, 1))
vector_for_char_a[char_to_ix['a']] = 1

#hyperparameters
hidden_size = 100
seq_length = 25 #25 chars generated every timestep
lr = 1e-1


#model
Wxh = np.random.randn(hidden_size, vocab_size) * 0.01 #input -> hidden
bh =  np.zeros((hidden_size, 1)) #hidden bias

Whh = np.random.randn(hidden_size, hidden_size) * 0.01 #hidden -> hidden

Why = np.random.randn(vocab_size, hidden_size) * 0.01 #hidden -> output
by =  np.zeros((vocab_size, 1)) #output bias

#memory of the neural network in each iteration
hprev = np.zeros((hidden_size, 1))

#smooth_loss = -np.log(1.0/vocab_size)*seq_length

#Training (trying to predict next character)
n,p = 0,0
ITERATIONS = 100000

#ADAMGRAD
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad 

all_loss = []

while n<= ITERATIONS:
    #when we finish the training data
    if p+seq_length+1 >= len(data) or n == 0:
        hprev = np.zeros((hidden_size, 1)) #FOR FORWARD PASS AT SAMPLE FUNCTION!!
        p = 0 #start again the training data
    
    #vetores de indices
    inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
    outputs = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

    #initializing vectors
    m = len(inputs)
    x, y = np.zeros((m, vocab_size,1)), np.zeros((m, vocab_size,1))
    z_1, a_1 = np.zeros((m, hidden_size,1)), np.zeros((m, hidden_size,1))
    #z_2, a_2 = np.zeros((m, vocab_size,1)), np.zeros((m, vocab_size,1))
    a_2 = np.zeros((m, vocab_size,1))
    a_1[-1] = np.copy(hprev) #copying the last state of the network in previou iteration

    loss = 0
    

    #forward pass
    for t in range(len(inputs)):

        x[t][inputs[t]] = 1
        y[t][outputs[t]] = 1

        z_1[t] = np.dot(Wxh, x[t]) + np.dot(Whh, a_1[t-1])# + bh
        a_1[t] = tanh(z_1[t])
        a_2[t] = softmax(np.dot(Why, a_1[t]))#+ by)
        #a_2[t] = softmax(z_2[t])

        #loss computation
        #loss -= np.log(a_2[t][outputs[t], 0])
        loss += -(np.log(a_2[t][outputs[t],0])) #a bit weird loss
        #loss -= (y[t]*np.log(a_2[t])).mean() #categorical cross entropy
        #loss -= (a_2[t]*np.log(a_2[t])).mean() #categorical cross entropy


    #loss /= len(inputs)
    #smooth_loss = smooth_loss * 0.999 + loss * 0.001
    #all_loss.append(smooth_loss)

    #gradients
    dWxh = np.zeros_like(Wxh)
    #dbh = np.zeros_like(bh)
    dWhh = np.zeros_like(Whh)
    dWhy = np.zeros_like(Why)
    #dby = np.zeros_like(by)
    hnext = np.zeros((hidden_size, vocab_size))

    #backward pass
    for t in range(len(inputs)):
        #hidden -> output layer
        dc_dz_o = (a_2[t] - y[t])/len(y[t])
        dWhy += np.dot(dc_dz_o,a_1[t].T)#/len(y[t])
        #dby += dc_dz_o


        #input -> hidden layer
        dc_da = np.dot(dc_dz_o.T, Why).T
        da_dw = np.dot(dtanh(z_1[t]),x[t].T) + dtanh(z_1[t])*np.dot(Whh, hnext)
        dWxh += dc_da*da_dw
        #dbh = dc_da *dtanh(z_1[t]) 

        hnext += da_dw #time update

        #hidden -> hidden layer
        dc_dz = np.dot(dc_dz_o.T, Why).T* dtanh(z_1[t])
        dz_dw = a_1[t-1].T if t!=0 else np.zeros_like(a_1[0]).T
        dWhh += np.dot(dc_dz, dz_dw)

        #exploding gradients solution
        for dparam in [dWxh, dWhh, dWhy]:#, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)


    hprev = a_1[-1]

    #sample from model to see the performance (just for showing off)
    if n % 1000 == 0:
        print(f'epoch: {n}, loss: {loss}') 

        #generate 200 characters to see how the network is
        x = np.zeros((vocab_size,1))
        x[inputs[0]] = 1
        indexes = []
        h = hprev
        for t in range(200):
            h = tanh(np.dot(Wxh, x) + np.dot(Whh, h))# + bh)
            predict = softmax(np.dot(Why, h))# + by)

            ix = np.random.choice(range(vocab_size), p=predict.ravel())

            #saving the predicted character 
            x = np.zeros((vocab_size,1))
            x[ix] = 1

            indexes.append(ix)

        txt = ''.join(ix_to_char[ix] for ix in indexes)
        print('----\n {} \n----'.format(txt))



    #optimizing parameters

    #for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
    #   [dWxh, dWhh, dWhy, dbh, dby],
    #   [mWxh, mWhh, mWhy, mbh, mby]):

    for param, dparam, mem in zip([Wxh, Whh, Why],
        [dWxh, dWhh, dWhy],
        [mWxh, mWhh, mWhy]):
#       
#
#       t = 0
#       m = [np.zeros_like(param[i]) for i in range(len(param))]
#       v = [np.zeros_like(param[i]) for i in range(len(param))]        
#       for i in range(len(m)):
#           t+=1
#           m[i] = b1 * m[i] + (1 - b1) * dparam[i] #estimates 1st momentum (mean) of gradient 
#           v[i] = b2 * v[i] + (1 - b2) * np.square(dparam[i]) #estimates 2nd momentum (variance) of gradient 
#           mhat = m[i] / (1. - b1**t) #corrects bias towards zero (initially set to vector of 0s, creates bias around it)
#           vhat = v[i] / (1. - b2**t)
#
#           param[i] -= lr*mhat / (np.sqrt(vhat) + eps) #updating params
#           

        mem += dparam * dparam
        param += -lr * dparam / np.sqrt(mem + 1e-8) # adagrad update


    p += seq_length # move data pointer
    n += 1      



import matplotlib.pyplot as plt
plt.plot(all_loss, label='loss')
plt.title('Loss During Training')
plt.show()  
