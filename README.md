# dervi8 - A painfully slow Machine Learning Framework

This is painfully slow Python Machine Learning implementation written from scratch for fun. It's not recomended to run
in production (or anywhere for that matter), unless you have a lot of time to kill.

## Features

* 2D Matrix implementation using `list`s of `list`s

## Requirements

* Python 3.8.5 of above (only tested on this)
* Poetry for managing dependencies

## Dataset

Using the MNIST dataset of images of hand-written digits from zero to nine.
http://yann.lecun.com/exdb/mnist/

## Optimized For Understanding

This code in the repo is optimized for understanding over performance. This is especially true when dealing with the
2-dimensions tensors (Tensor2D). Sometimes there's an obvious change that would make things go faster, but I wanted to
align it closer to what you would see in Numpy, PyTorch, Tensorflow or Pandas, which are optimized for handling large
multi-dimensional data structures and would cringe at my use of lists or lists of floats.

## Gradient Checking

Gradient checking is slow at the best of times and is exponentially slower here. To run gradient checking I reduced
layer units to a very small number and the mini-batch size to a small number.
