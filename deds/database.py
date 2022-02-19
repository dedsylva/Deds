import numpy as np
import pandas as pd
import math

class Wheat:
  def get_data(self, train=0.9):

    #loads data
    df = pd.read_csv('deds/datasets/seeds_dataset.csv')
    m = df.shape[0]
    n = df.shape[1]
    data = np.zeros((m,n))

    #to numpy
    for i in range(len(df)):
      data[i] = df.loc[i].to_numpy()


    #replacing possible 0,nans with mean of each feature
    for i in range(data.shape[1]):
      a = data[:,i]
      mean_i = a.mean()
      a = np.nan_to_num(a)
      a[a==0] = mean_i

    np.random.shuffle(data)
    train = math.ceil(train*m)

    X_train = data[:train,:-1].reshape(train, n-1, 1)
    X_test = data[train:,:-1].reshape(m-train,n-1, 1)
    Y_train = data[:train,-1].reshape(train,1)
    Y_test = data[train:,-1].reshape(m-train,1)

    return X_train, X_test, Y_train, Y_test

class MNIST:
  def get_data(self):
    #load data
    from deds.datasets import fetch_mnist
    from deds.extra.utils import to_categorical

    train_images, train_labels, test_images, test_labels = fetch_mnist()

    #need that channel dimension, normalized float32 tensor
    X_train = train_images.reshape((60000, 28*28, 1)).astype('float32')/255 
    Y_train =  to_categorical(train_labels).reshape((60000, 10, 1))
    X_test = test_images.reshape((10000, 28*28, 1)).astype('float32')/255 
    Y_test =  to_categorical(test_labels).reshape((10000, 10, 1))

    return X_train, X_test, Y_train, Y_test

class TTT:
  def get_data(self, data):
    chars = list(set(data)) #set filters unique characters already
    data_size, vocab_size = len(data), len(chars)

    # hot encoding (sparse, but just a example, should use word embedding)
    char_to_ix = { ch:i for i,ch in enumerate(chars)}
    ix_to_char = { i:ch for i,ch in enumerate(chars)}

    return vocab_size, char_to_ix, ix_to_char

