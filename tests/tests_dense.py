import numpy as np
import unittest
from unittest_prettify.colorize import colorize, GREEN 
from deds.model import Dense 
from deds.database import Wheat, MNIST 

@colorize(color=GREEN)
class TestMNIST(unittest.TestCase):
  lr = 0.001
  gamma = 0.95
  BS = 128
  db = MNIST()
  X_train, X_test, Y_train, Y_test = db.get_data()
  epochs = 20
  NN = Dense()
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

  def test_MNIST_training_data_shape(self):
    """ Test Shape of Training Data """
    self.assertEqual(self.X_train.shape, (60000, 28*28, 1))
    self.assertEqual(self.Y_train.shape, (60000, 10, 1))
  def test_MNIST_layers(self):
    """ Test Number of Layers is Compatible with Building Model """
    self.assertEqual(len(self.model), 3)
  def test_MNIST_loss(self):
    """ Test Loss Length and Range for MNIST """
    self.assertEqual(len(self.loss), self.epochs)
  def test_MNIST_acc(self):
    """ Test Accuracy for MNIST """
    self.assertEqual(len(self.accuracy), self.epochs)
    self.assertFalse(np.any(np.array(self.accuracy) > 1))
  def test_MNIST_weight_shapes(self):
    """ Test Weight Shapes of Model """
    self.assertEqual(len(self.model), 3)
    self.assertEqual(self.model[0][0].shape, (128, 28*28)) # shape of input
    self.assertEqual(self.model[0][1].shape, (128, 1)) # bias shape of input
    self.assertEqual(self.model[1][0].shape, (100, 128)) # weights of 1st hidden layer
    self.assertEqual(self.model[-1][0].shape[0], 10) # number of outputs
  def test_MNIST_activations(self):
    """ Test Activation Functions of Each Layer """
    self.assertEqual(self.model[0][2], 'ReLu')
    self.assertEqual(self.model[-1][2], 'Softmax')


@colorize(color=GREEN)
class TestWheat(unittest.TestCase): 
  db = Wheat()
  lr = 0.001
  BS = 8
  X_train, X_test, Y_train, Y_test = db.get_data(train=0.9)
  NN = Dense()
  epochs = 20
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

  def test_WHEAT_training_data_shape(self):
    """ Test Training Data for Wheat Example """
    self.assertEqual(self.X_train.shape, (189, 7, 1))
    self.assertEqual(self.Y_train.shape, (189, 1))
  def test_WHEAT_layers(self):
    """ Test Number of Layers is Compatible with Building Model """
    self.assertEqual(len(self.model), 3)

  def test_WHEAT_loss(self):
    """ Test Loss Length and Range for Wheat """
    self.assertEqual(len(self.loss), self.epochs)
  def test_WHEAT_acc(self):
    """ Test Accuracy for Wheat """
    self.assertEqual(len(self.accuracy), self.epochs)
    self.assertFalse(np.any(np.array(self.accuracy) > 1))
  def test_WHEAT_weight_shapes(self):
    """ Test Weight Shapes of Model """
    self.assertEqual(self.model[0][0].shape, (10, 7)) #weights shape of input
    self.assertEqual(self.model[0][1].shape, (10, 1)) #bias shape of input
    self.assertEqual(self.model[-1][0].shape[0], 1) #number of outputs
  def test_WHEAT_activations(self):
    """ Test Activation Functions of Each Layer """
    self.assertEqual(self.model[0][2], 'ReLu')
    self.assertEqual(self.model[1][2], 'ReLu')
    self.assertEqual(self.model[2][2], 'Linear')

if __name__ == '__main__':
  unittest.main(verbosity=2)     
