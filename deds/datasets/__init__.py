import numpy as np
import gzip

def fetch_mnist():
  parse = lambda file: np.frombuffer(gzip.open(file).read(), dtype=np.uint8).copy()
  X_train = parse("deds/datasets/mnist/train-images-idx3-ubyte.gz")[0x10:]
  Y_train = parse("deds/datasets/mnist/train-labels-idx1-ubyte.gz")[8:]
  X_test = parse("deds/datasets/mnist/t10k-images-idx3-ubyte.gz")[0x10:]
  Y_test = parse("deds/datasets/mnist/t10k-labels-idx1-ubyte.gz")[8:]
  return X_train, Y_train, X_test, Y_test
