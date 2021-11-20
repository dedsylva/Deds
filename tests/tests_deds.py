import numpy as np
import unittest
from unittest_prettify.colorize import colorize, GREEN
from deds.model import Dense, RNN 
from deds.database import Wheat, MNIST, TTT


@colorize(color=GREEN)
class TestMNIST(unittest.TestCase):
  def test_dataset(self):
    """ Testing Training Data for MNIST Dense Example """
    db = MNIST()
    X_train, X_test, Y_train, Y_test = db.get_data()
    assert X_train.shape == (60000, 28*28, 1), f'Invalid shape for X_train. Needs to be (60000, 28*28, 1), but instead got {X_train.shape}'
    assert Y_train.shape == (60000, 10, 1), f'Invalid shape for Y_train. Needs to be (60000, 10, 1), but instead got {Y_train.shape}'

  def test_model(self):
    """ Testing MNIST With Dense Model """
    db = MNIST()
    X_train, X_test, Y_train, Y_test = db.get_data()

    NN = Dense()
    epochs = 20
    lr = 0.001
    gamma = 0.95
    BS = 128
    model = NN.Input(128, input_shape=X_train.shape[1], activation='ReLu')      
    model = NN.Linear(128, 100, model, activation='ReLu')
    model = NN.Output(100, 10, model, activation='Softmax')

    #compile model
    NN.Compile(optimizer='SGD', loss='MSE', metrics='accuracy', lr=lr, 
                momentum=True, gamma=gamma)


    # train the model
    model, loss, accuracy = NN.Train(model, X_train, Y_train, 
            epochs=epochs, batch=BS, categoric=True, show_summary=False)


    # *** Testing *** #

    self.assertEqual(len(loss), epochs)
    self.assertEqual(len(accuracy), epochs)
    self.assertFalse(np.any(np.array(accuracy) > 1))
    self.assertEqual(len(model), 3)

    self.assertEqual(model[0][2], 'ReLu')
    self.assertEqual(model[0][0].shape, (128, X_train.shape[1])) # shape of input
    self.assertEqual(model[0][1].shape, (128, 1)) # bias shape of input

    self.assertEqual(model[1][0].shape, (100, 128)) # weights of 1st hidden layer

    self.assertEqual(model[-1][2], 'Softmax')
    self.assertEqual(model[-1][0].shape[0], 10) # number of outputs


@colorize(color=GREEN)
class TestWheat(unittest.TestCase): 
  def test_dataset(self):
    """ Testing Training Data for Wheat Example """
    db = Wheat()
    X_train, X_test, Y_train, Y_test = db.get_data(train=0.9)
    assert X_train.shape == (189, 7, 1) 
    assert Y_train.shape == (189, 1)

  def test_model(self):
    db = Wheat()
    X_train, X_test, Y_train, Y_test = db.get_data(train=0.9)

    NN = Dense()
    epochs = 20
    lr = 0.001
    BS = 8

    model = NN.Input(10, input_shape=X_train.shape[1], activation='ReLu')
    model = NN.Linear(10, 5, model, activation='ReLu')
    model = NN.Output(5, 1, model, activation='Linear')

    #compile model
    NN.Compile(optimizer='SGD', loss='MSE', metrics='accuracy', lr=lr, 
                momentum=True)


    #train the model
    model, loss, accuracy = NN.Train(model, X_train, Y_train, 
        epochs=epochs, batch=BS, categoric=False, show_summary=False)   


    # *** Testing *** #

    self.assertEqual(len(loss), epochs)
    self.assertEqual(len(accuracy), epochs)
    self.assertFalse(np.any(np.array(accuracy) > 1))
    self.assertEqual(len(model), 3)

    self.assertEqual(model[0][2], 'ReLu')
    self.assertEqual(model[0][0].shape, (10, 7)) #weights shape of input
    self.assertEqual(model[0][1].shape, (10, 1)) #bias shape of input

    self.assertEqual(model[-1][0].shape[0], 1) #number of outputs


@colorize(color=GREEN)
class TestSimpleRNN(unittest.TestCase):
  def test_model(self):
    """ Testing Simple RNN Model """

    source = 'deds/datasets/kafka.txt'
    with open(source, 'r', encoding='UTF-8') as d:
      data = d.read()

    db = TTT()
    vocab_size, char_to_ix, ix_to_char  = db.get_data(data)

    # hyperparameters
    epochs = 100
    lr = 0.01
    first_linear = 90
    hidden_size = 100
    linear_size = 90 
    seq_length = 25 # 25 chars generated every timestep
    NN = RNN()

    model = NN.Linear(vocab_size, first_linear, None, activation='ReLu')
    model = NN.RNN(first_linear, hidden_size, hidden_size, model)
    model = NN.Linear(hidden_size, linear_size, model, activation='ReLu')
    model = NN.Output(linear_size, vocab_size, model, activation='Softmax')

    # compile model
    NN.Compile(optimizer='SGD', loss='MSE', metrics='accuracy', 
               seq_length=seq_length, vocab_size=vocab_size, hidden_size=hidden_size, lr=lr, momentum=False)


    # train the model
    gradients, warning = NN.Train(model, data, 
                         char_to_ix, ix_to_char, epochs=epochs, show_summary=False, print_model=False, return_gradients=True) 


    # checking vanishing gradients
    grad_check_1_linear = np.zeros_like(model[0][0])
    grad_check_1_linear.fill(1e-02)

    grad_check_1_hidden = np.zeros_like(model[1][0])
    grad_check_1_hidden.fill(1e-02)

    grad_check_1_rnn = np.zeros_like(model[1][6])
    grad_check_1_rnn.fill(1e-02)

    grad_check_1_linear_2 = np.zeros_like(model[2][0])
    grad_check_1_linear_2.fill(1e-02)

    grad_check_1_output = np.zeros_like(model[3][0])
    grad_check_1_output.fill(1e-02)

    grad_check_2_linear = np.zeros_like(model[0][0]).fill(1e-04)
    grad_check_2_hidden = np.zeros_like(model[1][0]).fill(1e-04)
    grad_check_2_rnn = np.zeros_like(model[1][6]).fill(1e-04)
    grad_check_2_linear_2 = np.zeros_like(model[2][0]).fill(1e-04)
    grad_check_2_output = np.zeros_like(model[3][0]).fill(1e-04)

    grad_check_3_linear = np.zeros_like(model[0][0]).fill(1e-06)
    grad_check_3_hidden = np.zeros_like(model[1][0]).fill(1e-06)
    grad_check_3_rnn = np.zeros_like(model[1][6]).fill(1e-06)
    grad_check_3_linear_2 = np.zeros_like(model[2][0]).fill(1e-06)
    grad_check_3_output = np.zeros_like(model[3][0]).fill(1e-06)

    grad_check_4_linear = np.zeros_like(model[0][0]).fill(1e-09)
    grad_check_4_hidden = np.zeros_like(model[1][0]).fill(1e-09)
    grad_check_4_rnn = np.zeros_like(model[1][6]).fill(1e-09)
    grad_check_4_linear_2 = np.zeros_like(model[2][0]).fill(1e-09)
    grad_check_4_output = np.zeros_like(model[3][0]).fill(1e-09)

    # *** Testing *** #

    self.assertEqual(len(model), 4)

    self.assertEqual(model[0][2], 'ReLu')
    self.assertEqual(model[0][0].shape, (first_linear, vocab_size)) # weights shape of linear 

    self.assertEqual(model[1][2], 'Tanh') # activation of RNN layer is Tanh
    self.assertEqual(model[1][6].shape, (hidden_size, hidden_size)) # weights shape of hidden_cell 


    # Vanishing Gradients Test 
    # Difference of 1e-02 
    eps1=1e-1
    for gr in gradients:
      self.assertTrue(np.isclose(gr[0][0], grad_check_1_linear, atol=eps1).all()) 
      self.assertTrue(np.isclose(gr[1][0], grad_check_1_hidden, atol=eps1).all()) 
      self.assertTrue(np.isclose(gr[1][2], grad_check_1_rnn, atol=eps1).all()) 
      self.assertTrue(np.isclose(gr[2][0], grad_check_1_linear_2, atol=eps1).all()) 
      self.assertTrue(np.isclose(gr[3][0], grad_check_1_output, atol=eps1).all()) 

    # loss is too low
    self.assertFalse(warning) 

    self.assertEqual(model[-1][0].shape[0], vocab_size) # number of outputs
    self.assertEqual(model[-1][0].shape, (vocab_size, linear_size)) # weights shape of output 
    self.assertEqual(model[-1][2], 'Softmax') # activation of output


if __name__ == '__main__':
    unittest.main()     
