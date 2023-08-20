# TinyNN

English | [简体中文](README.zh-cn.md)

Build a Complete Neural Network in Less Than 2,000 Lines of Code

A minimized construction of neural network components. Supports both **Fully Connected Neural Network (FCNN)** and **Convolutional Neural Network (CNN)** for learning purposes.

## Installation

python version > 3.8

In your working directory, run: `python setup.py install`

## Tutorial Examples

In the `examples` directory, there are many sample codes:

```bash
├─examples
│      00-download_mnist.py                 # Download the mnist dataset
│      01-simple_forward.py                 # Understand the simplest forward computation
│      02-simple_backward.py                # Understand backward computation based on numerical differentiation
│      03-simple_network.py                 # Implement a two-layer neural network with both forward and backward computations
│      04-introduce_layer.py                # Introduce the concept of the layer
│      05-simple_network_with_layer.py      # Implement a neural network based on the layer concept
│      06-introduce_optimizer.py            # Introduce the concept of the optimizer and compare based on a specific function
│      07-introduce_optimizer_2.py          # Apply the optimizer in a real network
│      08-introduce_weight_decay.py         # Introduce the concept of weight decay
│      09-introduce_batch_normalization.py  # Introduce the concept of batch normalization
│      10-introduce_dropout.py              # Introduce the concept of dropout
│      11-hyperparam_search.py              # Implement hyperparameter search
│      12-CNN_and_digits_recognition.py     # Recognize handwritten digits using CNN
```

## References

- [pytorch](https://github.com/pytorch/pytorch)
- [tinynn](https://github.com/borgwang/tinynn)
- 《深度学习入门-基于Python的理论与实现》
