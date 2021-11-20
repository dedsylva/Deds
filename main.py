#! /usr/bin/env python

import sys
from deds.model import Dense, RNN 
from deds.database import Wheat, MNIST

def main(argv):
  args = [a.split('=') for a in argv]

  if args[0][0] == 'model':
    if args[0][1] == 'MNIST':
      db = MNIST()
      X_train, X_test, Y_train, Y_test = db.get_data()
      epochs = 100 
      BS = 128
      lr = 0.001
      gamma = 0.95

      NN = Dense()

      model = NN.Input(128, input_shape=X_train.shape[1], activation='ReLu')
      model = NN.Linear(128, 70, model, activation='ReLu')#, regularization='l2', reg=0.0001)
      model = NN.Dropout(model, p=0.5)
      model = NN.Output(70, 10, model, activation='Softmax')

      #compile model
      NN.Compile(optimizer='SGD', loss='MSE', metrics='accuracy', lr=lr, 
             momentum=True, gamma=gamma)

      #train the model
      model, losses, accuracy = NN.Train(model, X_train, Y_train, 
        epochs=epochs, batch=BS, categoric=True)

      #evaluate the network
      precision = NN.Evaluate(model, X_test, Y_test, True)

      import matplotlib.pyplot as plt
      plt.plot(range(epochs), accuracy, label='accuracy')
      plt.plot(range(epochs), losses, label='loss')
      plt.title('Trainning results')
      plt.legend()
      plt.show()

    elif args[0][1] == 'Wheat':
      db = Wheat()
      train = float(args[1][1]) if len(args) > 1 else 0.9
      X_train, X_test, Y_train, Y_test = db.get_data(train=train)
      epochs = 10000
      BS = 8
      lr = 0.001
      NN = Dense()

      model = NN.Input(10, input_shape=X_train.shape[1], activation='ReLu')
      model = NN.Linear(10, 5, model, activation='ReLu')
      model = NN.Output(5, 1, model, activation='Linear')

      #compile model
      NN.Compile(optimizer='SGD', loss='MSE', metrics='accuracy', lr=lr, 
             momentum=True)


      #train the model
      model, losses, accuracy = NN.Train(model, X_train, Y_train, 
        epochs=epochs, batch=BS, categoric=False) 

      #evaluate the network
      precision = NN.Evaluate(model, X_test, Y_test, False)

      import matplotlib.pyplot as plt
      plt.plot(range(epochs), accuracy, label='accuracy')
      plt.plot(range(epochs), losses, label='loss')
      plt.title('Trainning results')
      plt.legend()
      plt.show()

    elif args[0][1] == 'RNN':
      print_model = args[1][1] if ( len(args) > 1 and args[1][0] in('print', 'Print'))  else False

      # source = 'harry_potter.txt'
      source = 'deds/datasets/kafka.txt'
      data = open(source, 'r', encoding='UTF-8').read()
      chars = list(set(data)) #set filters unique characters already
      data_size, vocab_size = len(data), len(chars)

      #X_train, X_test, Y_train, Y_test = db.get_data(train=train)

      # hot encoding (sparse, but just a example, should use word embedding)
      char_to_ix = { ch:i for i,ch in enumerate(chars)}
      ix_to_char = { i:ch for i,ch in enumerate(chars)}


      # hyperparameters
      epochs = 10000
      BS = 8
      lr = 0.01
      first_linear = 90
      hidden_size = 150
      hidden_size_2 = 100
      linear_size = 120
      seq_length = 25 # 25 chars generated every timestep
      NN = RNN()

      #model = NN.Input(hidden_size, input_shape=(vocab_size,), activation='ReLu')
      model = NN.Linear(vocab_size, first_linear, None, activation='ReLu')
      #model= NN.RNN(vocab_size, hidden_size, hidden_size, None)
      model = NN.RNN(first_linear, hidden_size, hidden_size, model)
      model = NN.RNN(hidden_size, hidden_size_2, hidden_size_2, model)
      model = NN.Linear(hidden_size_2, linear_size, model, activation='ReLu')
      model = NN.Output(linear_size, vocab_size, model, activation='Softmax')

      #compile model
      NN.Compile(optimizer='SGD', loss='MSE', metrics='accuracy', 
                 seq_length=seq_length, vocab_size=vocab_size, hidden_size=hidden_size, lr=lr, momentum=False)


      #train the model
      NN.Train(model, data, 
        char_to_ix, ix_to_char, epochs=epochs, batch=BS, print_model=False) 

    else:
      raise Exception ('The Model you entered are not available')
  else:
    raise ValueError ('The argument \'{}\' is invalid!'.format(args[0][0]))

if __name__ == '__main__':
  main(sys.argv[1:])
