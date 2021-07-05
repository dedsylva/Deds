# Deds - Deep Learning From Scratch

![](https://img.shields.io/badge/tests-passing-green)

## Objective
Your simple numpy from scratch deep learning library, created just to keep track of what's happening behind the scenes of TensorFlow. The implementation is similar to Keras. Currently there's only available the Dense format with a few simple functions and the SGD (Stochastic Gradient Descent) optimizer.

## Implementation
For a simple test

```python
python deds.py model=MNIST #mnist
python deds.py model=Wheat #wheat seeds class prediction
```

Or just modify using <b>main.py</b>

## Example at Deds
```python
# input_shape must have (batch, features, 1)
# unlike keras that goes (batch, features, )
from model import Model
NN = Model()

model = NN.Input(15, input_shape=X_train.shape[1], activation='ReLu')
model = NN.Dense(15, 10, model, activation='ReLu')
model = NN.Output(10, 1, model, activation='Linear')

#train the model categoric labels for now needs to be explicit
loss, accuracy = NN.Train(model, X_train, Y_train, 
	loss='MSE', opt='SGD', epochs=10, batch=64, categoric=False, lr=0.001, gamma = 0.95)
```

## Example at Keras
```python
from keras import layers, models

model = models.Sequential()
model.add(layers.Dense(15, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='linear'))

model.compile(optimizer='sgd', loss='mse', metrics= ['accuracy'])

history = model.fit(
    x=X_train, y=Y_train, 
    epochs=10, batch_size=8
)

```


## Tests

```python
python -m unittest tests/tests_deds.py
```

### Todo
* Implement Dense 
 * Implement forward/backward pass -- done
 * Implement SGD Optimizer -- done
 * Implement Momentum -- done
 * Implement L2 Regularization
 * Implement Dropout
 * Implement Adam optimizer
* Implement gradcheck (numeric finite difference gradient calc)
* Implement convolutions
* Implement LSTM