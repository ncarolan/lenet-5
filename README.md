# LeNet-5

![lenet5](src/res/LeCun98.png)
<p align="center"><sub>[<a href="http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf">Image Source</a>]</sub></p>

This repository implements the LeNet-5 model for handwritten digit recognition. The implementation is based on the model described in *"Gradient-Based Learning Applied to Document Recognition"* (1998) by LeCun, Bottou, Bengio, and Haffner. 

Some architectural changes have been made to the 1998 model. Notably, that model took 32x32 pixel inputs; however the modern MNIST dataset used in this repository has 28x28 pixel image dimensions. 

### Usage
Models can be trained and evaluated with [`lenet.ipynb`](lenet.ipynb). [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ncarolan/lenet-5/blob/main/lenet.ipynb)

You can also train directly from the command line by running `$ python src/train.py`.

### Ablations

### Interpretability
