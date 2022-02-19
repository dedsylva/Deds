# New ways to initialize the Model (Dense only for now)

```python
import numpy as np
import deds.activation
from deds.model import Dense

x = np.random.rand(784,1)

NN = Dense.Input(128, input_shape=784, activation=deds.activation.ReLu) # Returns your model
forw_1 = Dense.forward(NN[-1], x) # Returns the forward pass (input, z, act(z))

# Adding more Layers
NN = Dense.Linear(128, 70, NN, activation=deds.activation.ReLu)
forw_2 = Dense.forward(NN[-1], forw_1[2]) # Returns the forward pass of that layer

```

## What can we do with that
* Test if deds.activation Functions is called in the forward/backward pass
* Test the gradients with the backward pass
* Train our model the way we want (without the .train method)
