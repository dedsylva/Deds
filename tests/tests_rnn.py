import numpy as np
from deds.model import RNN
import unittest
from unittest_prettify.colorize import colorize, GREEN 
from deds.database import TTT

@colorize(color=GREEN)
class TestSimpleRNN(unittest.TestCase):
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
  loss, gradients, warning = NN.Train(model, data, 
                       char_to_ix, ix_to_char, epochs=epochs, show_summary=False, print_model=False, return_gradients=True) 


    # *** Testing *** #

  def test_RNN_layers(self):
    """ Test Number of Layers is Compatible with Building Model """
    self.assertEqual(len(self.model), 4)
  def test_RNN_loss(self):
    """ Test Loss Length and Range for Wheat """
    self.assertEqual(len(self.loss), self.epochs)
  def test_RNN_loss_too_low(self):
    """ Test to see if loss is getting too low during training """
    self.assertFalse(self.warning) 
  def test_RNN_weight_shapes(self):
    """ Test Weight Shapes of Model """
    self.assertEqual(self.model[0][0].shape, (self.first_linear, self.vocab_size)) # weights shape of first linear layer 
    self.assertEqual(self.model[1][0].shape, (self.hidden_size, self.first_linear)) # weights shape of RNN layer 
    self.assertEqual(self.model[1][6].shape, (self.hidden_size, self.hidden_size)) # weights shape of cell of RNN layer 
    self.assertEqual(self.model[2][0].shape, (self.linear_size, self.hidden_size)) # weights shape of second linear layer 
    self.assertEqual(self.model[3][0].shape, (self.vocab_size, self.linear_size)) # weights shape of output 
  def test_RNN_cell_state_activation(self):
    """ Test if RNN Layer has Tanh Activation """
    self.assertEqual(self.model[1][2], 'Tanh') # activation of RNN layer is Tanh
  def test_RNN_output_activation(self):
    """ Test Output Activation Function """
    self.assertEqual(self.model[-1][2], 'Softmax') # activation of output

  def test_RNN_vanshing_gradients(self):
    """ Test if Gradients are not Vanishing """

    # checking vanishing gradients
    grad_check_1_linear = np.zeros_like(self.model[0][0])
    grad_check_1_linear.fill(1e-02)

    grad_check_1_hidden = np.zeros_like(self.model[1][0])
    grad_check_1_hidden.fill(1e-02)

    grad_check_1_rnn = np.zeros_like(self.model[1][6])
    grad_check_1_rnn.fill(1e-02)

    grad_check_1_linear_2 = np.zeros_like(self.model[2][0])
    grad_check_1_linear_2.fill(1e-02)

    grad_check_1_output = np.zeros_like(self.model[3][0])
    grad_check_1_output.fill(1e-02)

#4
    grad_check_2_linear = np.zeros_like(self.model[0][0])
    grad_check_2_linear.fill(1e-04)

    grad_check_2_hidden = np.zeros_like(self.model[1][0])
    grad_check_2_hidden.fill(1e-04)

    grad_check_2_rnn = np.zeros_like(self.model[1][6])
    grad_check_2_rnn.fill(1e-04)

    grad_check_2_linear_2 = np.zeros_like(self.model[2][0])
    grad_check_2_linear_2.fill(1e-04)

    grad_check_2_output = np.zeros_like(self.model[3][0])
    grad_check_2_output.fill(1e-04)

#6
    grad_check_3_linear = np.zeros_like(self.model[0][0])
    grad_check_3_linear.fill(1e-06)

    grad_check_3_hidden = np.zeros_like(self.model[1][0])
    grad_check_3_hidden.fill(1e-06)

    grad_check_3_rnn = np.zeros_like(self.model[1][6])
    grad_check_3_rnn.fill(1e-06)

    grad_check_3_linear_2 = np.zeros_like(self.model[2][0])
    grad_check_3_linear_2.fill(1e-06)

    grad_check_3_output = np.zeros_like(self.model[3][0])
    grad_check_3_output.fill(1e-06)


#9
    grad_check_4_linear = np.zeros_like(self.model[0][0])
    grad_check_4_linear.fill(1e-10)

    grad_check_4_hidden = np.zeros_like(self.model[1][0])
    grad_check_4_hidden.fill(1e-10)

    grad_check_4_rnn = np.zeros_like(self.model[1][6])
    grad_check_4_rnn.fill(1e-10)

    grad_check_4_linear_2 = np.zeros_like(self.model[2][0])
    grad_check_4_linear_2.fill(1e-10)

    grad_check_4_output = np.zeros_like(self.model[3][0])
    grad_check_4_output.fill(1e-10)


    # *** Testing *** #
    # Vanishing Gradients Test 
    eps1=1e-1
    # Difference of 1e-02 
    for gr in self.gradients:
      self.assertTrue(np.isclose(gr[0][0], grad_check_1_linear, atol=eps1).all()) 
      self.assertTrue(np.isclose(gr[1][0], grad_check_1_hidden, atol=eps1).all()) 
      self.assertTrue(np.isclose(gr[1][2], grad_check_1_rnn, atol=eps1).all()) 
      self.assertTrue(np.isclose(gr[2][0], grad_check_1_linear_2, atol=eps1).all()) 
      self.assertTrue(np.isclose(gr[3][0], grad_check_1_output, atol=eps1).all()) 

    # Difference of 1e-04 
    eps2=1e-2
    for gr in self.gradients:
      self.assertTrue(np.isclose(gr[0][0], grad_check_2_linear, atol=eps2).all()) 
      self.assertTrue(np.isclose(gr[1][0], grad_check_2_hidden, atol=eps2).all()) 
      self.assertTrue(np.isclose(gr[1][2], grad_check_2_rnn, atol=eps2).all()) 
      self.assertTrue(np.isclose(gr[2][0], grad_check_2_linear_2, atol=eps2).all()) 
      self.assertTrue(np.isclose(gr[3][0], grad_check_2_output, atol=eps2).all()) 

    # Difference of 1e-06 
    eps3=1e-19
    for gr in self.gradients:
      self.assertFalse(np.isclose(gr[0][0], grad_check_3_linear, atol=eps3).all()) 
      self.assertFalse(np.isclose(gr[1][0], grad_check_3_hidden, atol=eps3).all()) 
      self.assertFalse(np.isclose(gr[1][2], grad_check_3_rnn, atol=eps3).all()) 
      self.assertFalse(np.isclose(gr[2][0], grad_check_3_linear_2, atol=eps3).all()) 
      self.assertFalse(np.isclose(gr[3][0], grad_check_3_output, atol=eps3).all()) 

    # Difference of 1e-09 
    eps4=1e-12
    for gr in self.gradients:
      self.assertFalse(np.isclose(gr[0][0], grad_check_4_linear, atol=eps4).all()) 
      self.assertFalse(np.isclose(gr[1][0], grad_check_4_hidden, atol=eps4).all()) 
      self.assertFalse(np.isclose(gr[1][2], grad_check_4_rnn, atol=eps4).all()) 
      self.assertFalse(np.isclose(gr[2][0], grad_check_4_linear_2, atol=eps4).all()) 
      self.assertFalse(np.isclose(gr[3][0], grad_check_4_output, atol=eps4).all()) 






if __name__ == '__main__':
  unittest.main(verbosity=2)
