#! /usr/bin/env python

import sys
import os
from deds.model import Dense, RNN 
from deds.database import Wheat, MNIST, TTT 
from deds.extra.utils import fetch 

def main(argv):
  _model = os.environ.get('MODEL')

  if _model == 'MNIST':
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

  elif _model == 'WHEAT':
    db = Wheat()
    train = 0.9
    X_train, X_test, Y_train, Y_test = db.get_data(train=train)
    epochs = 1000
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

  elif _model == 'RNN':
    print_model = bool(os.environ.get('PRINT')) if os.environ.get('PRINT') is not None else False
    _source = argv[0] if len(argv) > 0 else 'deds/datasets/kafka.txt'

    if _source.startswith('http'):
      data = fetch(_source)
    elif _source == 'hp':
      data = 'deds/datasets/harry_potter.txt'
      with open(source, 'r', encoding='UTF-8') as d:
        data = d.read()
    else:
      source = 'deds/datasets/kafka.txt'
      with open(source, 'r', encoding='UTF-8') as d:
        data = d.read()

    db = TTT()
    vocab_size, char_to_ix, ix_to_char  = db.get_data(data)

    # hyperparameters
    epochs = 10000
    lr = 0.01
    first_linear = 90
    hidden_size = 150
    hidden_size_2 = 100
    linear_size = 120
    seq_length = 25 # 25 chars generated every timestep
    NN = RNN()

    model = NN.Linear(vocab_size, first_linear, None, activation='ReLu')
    model = NN.RNN(first_linear, hidden_size, hidden_size, model)
    model = NN.RNN(hidden_size, hidden_size_2, hidden_size_2, model)
    model = NN.Linear(hidden_size_2, linear_size, model, activation='ReLu')
    model = NN.Output(linear_size, vocab_size, model, activation='Softmax')

    #compile model
    NN.Compile(optimizer='SGD', loss='MSE', metrics='accuracy', 
               seq_length=seq_length, vocab_size=vocab_size, hidden_size=hidden_size, lr=lr, momentum=False)


    #train the model
    loss, warning = NN.Train(model, data, 
      char_to_ix, ix_to_char, epochs=epochs, print_model=print_model) 


  else:
      raise Exception ('The Model you entered are not available')


if __name__ == '__main__':
  main(sys.argv[1:])
