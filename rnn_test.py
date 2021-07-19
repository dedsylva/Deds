import numpy as np

def sigmoid(x):
	return 1/(1 + np.exp(-x))

def tanh(x):
	return np.tanh(x)	

def Softmax(x):
	max_ = np.max(x)
	return np.exp(x-max_)/sum(np.exp(x-max_))

def loss(inputs, targets, hprev):
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
		hs[t] = tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) #hidden state
		ys[t] = np.dot(Why, hs[t]) + by #unnormalized log probabilities for next chars
		ps[t] = softmax(ys[t]) #
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

		dWhy += np.dot(dy, hs[t].T)
		dby += dy


data = open('kafka.txt', 'r').read()
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
lr = 1e-1

#model parameters
Wxh = np.random.randn(hidden_size, vocab_size) * 0.01 #input to hidden
bh = np.zeros((hidden_size, 1)) #hidden bias

Whh = np.random.randn(hidden_size, hidden_size) * 0.01 #hidden to hidden

Why = np.random.randn(hidden_size, vocab_size) * 0.01 #hidden to output
bh = np.zeros((vocab_size, 1)) #output bias

#forward pass
#h[t] = actv(Wx[t] + Uh[t-1])
