import numpy as np

def sigmoid(x):
	return 1/(1 + np.exp(-x))

def tanh(x):
	return np.tanh(x)	

def softmax(x):
	max_ = np.max(x)
	return np.exp(x-max_)/sum(np.exp(x-max_))

def lossFun(inputs, targets, hprev):
	#xs = input one hot encoded chars
	#hs = hidden state outputs
	#ys = target values
	#ps = normalized probabilities of ys (softmax)
	xs, hs, ys, ps = {}, {}, {}, {}

	hs[-1] = np.copy(hprev)
	#loss init to 0
	loss = 0

	#forward pass
	for t in range(len(inputs)):
		xs[t] = np.zeros((vocab_size, 1))
		xs[t][inputs[t]] = 1 
		hs[t] = tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # a_1[t] hidden state
		ys[t] = np.dot(Why, hs[t]) + by # z_2[t]unnormalized log probabilities for next chars
		ps[t] = softmax(ys[t]) #a_2[t]

		loss += -np.log(ps[t][targets[t],0])

	#backward pass
	dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)	
	dbh, dby = np.zeros_like(bh), np.zeros_like(by)	
	dhnext = np.zeros_like(hs[0])

	for t in reversed(range(len(inputs))):
		#output probabilities
		dy = np.copy(ps[t])
		dy[targets[t]] -= 1 #NAO ENTENDI, NAO CONCORDO
		#dy = - np.copy(ps[t]) -- PRA MIM DEVERIA SER ISSO (dc_dz)

		#output layer
		dWhy += np.dot(dy, hs[t].T)
		dby += dy

		#hidden layer
		dh = np.dot(Why.T, dy) + dhnext #dc_da_1 esse ultimo termo n ta claro, deve ser algo da backprogragation com tempo
		#(1 - hs[t] * hs[t]) da_1_dz_1
		dhraw = (1 - hs[t] * hs[t]) * dh #dc_dz_1
		dbh += dhraw #db1
		dWxh += np.dot(dhraw, xs[t].T) #dw1
		dWhh += np.dot(dhraw, hs[t-1].T) #novo termo devido ao novo weight
		dhnext = np.dot(Whh.T, dhraw)

	#exploding gradients solution
	for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
		np.clip(dparam, -5, 5, out=dparam)

	return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]


#prediction, one full forward pass
def sample(h, seed_ix, n):
	x = np.zeros((vocab_size, 1))
	x[seed_ix] = 1
	ixes = [] # list to store generated chars

	for t in range(n):
		#forward pass
		h = tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
		y = np.dot(Why, h) + by

		#taken probabilities of next char
		p = softmax(y)

		#pick the one with highest prob
		ix = np.random.choice(range(vocab_size), p=p.ravel())

		x = np.zeros((vocab_size, 1))
		x[ix] = 1
		ixes.append(ix)

	txt = ''.join(ix_to_char[ix] for ix in ixes)
	print('----\n {} \n----'.format(txt))

#source = 'harry_potter.txt'
source = 'kafka.txt'
data = open(source, 'r', encoding='UTF-8').read()
chars = list(set(data))
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
learning_rate = 1e-1

#model parameters
Wxh = np.random.randn(hidden_size, vocab_size) * 0.01 #input to hidden
bh = np.zeros((hidden_size, 1)) #hidden bias

Whh = np.random.randn(hidden_size, hidden_size) * 0.01 #hidden to hidden

Why = np.random.randn(vocab_size, hidden_size) * 0.01 #hidden to output
by = np.zeros((vocab_size, 1)) #output bias

hprev = np.zeros((hidden_size,1)) # reset RNN memory  
#predict the 200 next characters given 'a'
sample(hprev,char_to_ix['a'],200)

#Training
p=0  
inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad 

smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
all_loss = []

while n<=1000*100: 
	# prepare inputs (we're sweeping from left to right in steps seq_length long)
	# check "How to feed the loss function to see how this part works
	if p+seq_length+1 >= len(data) or n == 0:
		hprev = np.zeros((hidden_size,1)) # reset RNN memory
		p = 0 # go from start of data
	inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
	targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

	# forward seq_length characters through the net and fetch gradient
	loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
	smooth_loss = smooth_loss * 0.999 + loss * 0.001
	all_loss.append(smooth_loss)
	 # sample from the model now and then
	if n % 1000 == 0:
	 	print('iter %d, loss: %f' % (n, smooth_loss)) # print progress
	 	sample(hprev, inputs[0], 200)

	# perform parameter update with Adagrad  
	for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
	 	[dWxh, dWhh, dWhy, dbh, dby],
	 	[mWxh, mWhh, mWhy, mbh, mby]):

		mem += dparam * dparam
		param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

	p += seq_length # move data pointer
	n += 1	 	


import matplotlib.pyplot as plt
plt.plot(all_loss, label='loss')
plt.title('Loss during Training')	
plt.show()
