# 2-Layer Deep Neural Network With No Dependencies

This is a "from scratch" Machine Learning implementation with no dependencies except for Python and pytest. So obviously
 a bit slower than using highly optimized Tensorflow, Pytorch, numpy or pandas.

Why? I wrote this for fun, but also to ensure I understood the lower-level details of implementating a Deep Neural
Network.

It's not recommended to run this in production, or anywhere for that matter, unless you have a lot of time to kill.
Surprisingly though, it was not as slow as I was expecting.

Training uses the [MNIST Database of Handwritten Digits](http://yann.lecun.com/exdb/mnist/).

![](example-images.png)

_[Image by Josef Steppan - CC BY-SA 4.0](https://en.wikipedia.org/wiki/MNIST_database#/media/File:MnistExamples.png)_

## Features

* Input layer for the 28x28 grayscale pixel images
* 2 full-connected 32 unit hidden layers with ReLU activation
* Output layer using Softmax activation to 10 classes (0-9)
* Mini-batch Gradient Descent
* Back propagation
* Gradient checking
* Input data normalization (0 to 255 => 0.0 to 0.1)
* 2D Tensor implementation using `list`s of `list`s!
* Broadcasting

## Requirements

* Python 3.8.5 or above (only tested on this)
* Poetry for managing dependencies

## Dataset

Using the MNIST dataset of images of hand-written digits from zero to nine.
http://yann.lecun.com/exdb/mnist/

This consists 28 x 28 grayscale (0-255) pixels images and corresponding output labels (0-9).

60,000 training examples. 10,000 test examples. Luckily this all fit in memory.

Used a mini-batch size of 500.

## Setup

Install the dependencies. Although, since this is just Pytest, only really needed if you're running the tests.
```
poetry install
```

Fetch the dataset from http://yann.lecun.com/exdb/mnist/ and store them in `datasets/mnist/`.
This will download four gzipped files.
Do not unzip them, since the code reads the zipped version directly.
```
make datasets
```

## Results

Some results are listed in the [results/](results/) directory, but generally seeing over +90% test accuracy after
6 epochs and maxing out a few percent higher. I haven't run it exhaustively yet.

Each epoch takes roughly 16-17 minutes to run. This better than I expected, since it's going through 60,000
images and using rudimentary data types.

Regularization didn't seem to be needed. I think this due to not running it exhaustively. Although, it may have
slightly improved training speed (accuracy seem to rise a little quicker). Fairly negligible and this theory may
have been disproved with more training runs.

## Hardware

This was run on the following hardware...

* MacBook Pro (16-inch, 2019)
* 2.4 GHz 8-Core Intel Core i9
* 32 GB 2667 MHz DDR4

It also has a GPU, but wasn't utilized.

No parallelization was implemented in the Python code.

## Optimized For Understanding

This code in the repo has been optimized for understanding over performance. This is especially true when dealing with
the 2-dimensions tensors ("`Tensor2D`" which is a `list` of `list` of `float`). Sometimes there's an obvious change that
would make things go faster, but I didn't want it to become less readable and wanted it to align closer to what you
would see in Numpy, PyTorch, Tensorflow or Pandas, which are optimized for handling large multi-dimensional data
structures and would cringe at my use of lists or lists of floats.

## Gradient Checking

Gradient checking is slow at the best of times and is exponentially slower here. To run gradient checking I reduced
layers to a very small number of units and reduced the mini-batch size to a small number.

## Learnings

### Things can look good, when they ain't

I found the initial implementation appeared to work. Things were running and the loss was going down!
Unfortunately, the accuracy was stuck bouncing around 10%. I still had a few hard-to-debug issues.

### Gradient checking FTW!

Gradient checking helped see where some issues were. I was able to check gradients for different groups of parameters
and work backwards through the network to try and understand where the issues were.

### ReLU derivative - simple?

A few things with the ReLU tripped me up. When the derivative is zero and the gradient is killed, it confused me a
little. Other issues seemed to compound this, so I was seeing a lot of zeros.

At one point I had the [ReLU derivative always returning one](https://github.com/philwhln/deriv8/commit/a9552a970).
The simplest issues can cause the biggest problems and this was one of the few things I didn't have tests for.

### Jacobian Matrices are not required for Softmax backprop

Luckily, [Jacobian Matrices](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant) are rarely required and
they often reduce to something simpler.

I learnt this the hard way, [implementing a Jacobian matrice](https://github.com/philwhln/deriv8/commit/1794bf36da) for
the Softmax derivative and trying to multiply that by the derivative of the loss function. It was getting complicated,
so I assumed I was doing something wrong and discovered I was. The product of the loss and softmax derivatives result
is simply `Y_hat - Y`!

Shortly afterwards, Andrej Karpathy confirmed this for me on YouTube...

_["You never end up forming the full Jacobian"](https://youtu.be/i94OvYb6noo?t=2668) - Andrej Karpathy_

_"Doh!" - Me_

Later I went back to fully understand the derivation and can recommend this two-part explanation from
[Mehran Bazargani](https://twitter.com/MLDawn2018)...

* [What is the derivative of the Softmax Function?](https://www.youtube.com/watch?v=09c7bkxpv9I)
* [Back propagation through Cross Entropy and Softmax](https://www.youtube.com/watch?v=znqbtL0fRA0)

## Credits

These two courses were probably the most helpful...

**Machine Learning - Andrew Ng**

https://www.coursera.org/learn/machine-learning

**Deep Learning Specialization - Andrew Ng**

https://www.coursera.org/specializations/deep-learning
