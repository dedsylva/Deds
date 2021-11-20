# Deds - Deep Learning From Scratch

![](https://img.shields.io/badge/tests-passing-green)

## Objective
Your simple numpy from scratch deep learning library, created just to keep track of what's happening behind the scenes of TensorFlow. The implementation is similar to Keras. Currently there's only available the Dense format with a few simple functions and the SGD (Stochastic Gradient Descent) optimizer.

## Implementation
For a simple test

```python
python main.py model=MNIST # mnist
python main.py model=Wheat # wheat seeds class prediction
python main.py model=RNN print=True # simple RNN model trying to reproduce text input (source)
```

Or just modify using <b>main.py</b>

## Example at Deds
```python
# input_shape must have (batch, features, 1)
# unlike keras that goes (batch, features, )
from model import Dense 
NN = Dense()

model = NN.Input(128, input_shape=X_train.shape[1], activation='ReLu')
model = NN.Linear(128, 70, model, activation='ReLu', regularization='l2', reg=0.00001) #a little low, I know 
model = NN.Output(70, 10, model, activation='Softmax', regularization='l1', reg=0.0001)

NN.Compile(optimizer='SGD', loss='MSE', metrics='accuracy', lr=lr, gamma=gamma)

model, loss, accuracy = NN.Train(model, X_train, Y_train, 
                    epochs=20, batch=128, categoric=True)

precision = NN.Evaluate(model, X_test, Y_test, categoric=True)
```

## Example at Keras
```python
from keras import layers, models

model = models.Sequential()
model.add(layers.Dense(15, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(10, kernel_regularizer=regularizers.l2(0.02), activation='relu'))
model.add(layers.Dense(1, kernel_regularizer=regularizers.l2(0.02), activation='linear'))

model.compile(optimizer='sgd', loss='mse', metrics= ['accuracy'])

history = model.fit(
    x=X_train, y=Y_train, 
    epochs=10, batch_size=8
)

model.evaluate(X_test, Y_test)

```


## Tests

```python
python -m unittest tests/tests_deds.py
```

### Todo
<b>* Tests for RNN </b>
  * Put Dense as the first layer, then RNN
  * Support L1, L2 Regularizations for Dense Layers in RNN Class
  * Creating Multiple RNNs with different hidden sizes between them

<b>* Clean RNN code (optimizers and Train)</b>

* Dense 
    * Implement forward/backward pass [x]
    * Implement SGD Optimizer [x]
    * Implement Momentum [x]
    * Implement L1 Regularization [x]
    * Implement L2 Regularization [x]
    * Implement Dropout -- failing
    * Implement Adam optimizer [x]
* Implement gradcheck (numeric finite difference gradient calc)
* Convolutions
* RNN 
    * Implement Vanilla RNN [] 
    * Implement LSTM []
* Wrap Models into Sequential Class 
* Implement Functional Class 
