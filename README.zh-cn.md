# TinyNN

[English](README.md) | 简体中文

使用不到两千行代码构建一个完整的神经网络

最小化构建的神经网络运行组件，支持**全连接神经网络（FCNN）**和**卷积神经网络（CNN）**，用于学习目的。

## 安装

python version > 3.8

在工作目录下，`python setup.py install`

## 教程示例

在`examples`目录下有许多示例代码和对应注解

```bash
├─examples
│      00-download_mnist.py                 # 下载mnist数据集
│      01-simple_forward.py                 # 理解最简单的前向计算
│      02-simple_backward.py                # 基于数值微分理解反向计算
│      03-simple_network.py                 # 实现包括前向和反向的两层神经网络
│      04-introduce_layer.py                # 引入layer层的概念
│      05-simple_network_with_layer.py      # 基于layer实现神经网络
│      06-introduce_optimizer.py            # 引入优化器的概念，并基于一个特定函数比较
│      07-introduce_optimizer_2.py          # 在实际网络中运用优化器
│      08-introduce_weight_decay.py         # 引入权值衰减的概念
│      09-introduce_batch_normalization.py  # 引入批标准化的概念
│      10-introduce_dropout.py              # 引入dropout的概念
│      11-hyperparam_search.py              # 实现超参数搜索
│      12-CNN_and_digits_recognition.py     # 基于CNN完成手写数字识别过程
```

## 参考

- [pytorch](https://github.com/pytorch/pytorch)
- [tinynn](https://github.com/borgwang/tinynn)
- 《深度学习入门-基于Python的理论与实现》
