'''
Implementations of the LeNet-5 image classification model in JAX / FLAX.
'''

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state

class FlaxLeNet(nn.Module):
    """ LeNet-5 in Flax adapted to 28x28 pixel inputs (modern MNIST).  """

    def __call__(self, x):
        x = nn.Conv(features=6, kernel_size=(5,5), padding='SAME')(x)
        x = nn.sigmoid(x)
        x = nn.MaxPool2d(x, window_shape=(2,2), strides=(2,2))

        x = nn.Conv(feature=16, kernel_size=(5,5))(x)
        x = nn.sigmoid(x)
        x = nn.MaxPool2d(x, window_shape=(2,2), strides=(2,2))

        x = x.reshape(x.shape[0], -1) # Flatten all dimensions except batch
        x = nn.Dense(features=120)(x)
        x = nn.sigmoid(x)

        x = nn.Dense(features=84)(x)
        x = nn.sigmoid(x)

        x = nn.Dense(features=10)(x)
        return x


class JaxLeNet():
     """ LeNet-5 in pure JAX adapted to 28x28 pixel inputs (modern MNIST).  """

     def relu(x):
        return jnp.maximum(0,x)
