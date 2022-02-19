#! usr/bin/python3

from enum import Enum, unique

@unique
class Types(Enum):
  Input = 'Input'
  Output = 'Output'
  Linear = 'Linear'
  RNN = 'RNN'
  Dropout = 'Dropout'
  LSTM = 'LSTM'

@unique
class Regs(Enum):
  L1 = 'L1'
  L2 = 'L2'
  No = None


import numpy as np

# Keras function without having to pip3 install tensorflow (500 MB is A LOT)
def to_categorical(y, num_classes=None, dtype='float32'):

  y = np.array(y, dtype='int')
  input_shape = y.shape
  if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
    input_shape = tuple(input_shape[:-1])
  y = y.ravel()
  if not num_classes:
    num_classes = np.max(y) + 1
  n = y.shape[0]
  categorical = np.zeros((n, num_classes), dtype=dtype)
  categorical[np.arange(n), y] = 1
  output_shape = input_shape + (num_classes,)
  categorical = np.reshape(categorical, output_shape)
  return categorical


def fetch(url):
  import requests, os, hashlib, tempfile
  fp = os.path.join(tempfile.gettempdir(), hashlib.md5(url.encode('utf-8')).hexdigest()) # creates /tmp/<hexdecimal> directory
  print("fething %s", url)
  dat = requests.get(url).content
  with open(fp+".tmp", "wb") as f:
    f.write(dat)
  os.rename(fp+".tmp", fp)
  return dat


