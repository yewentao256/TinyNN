from tinynn.mnist import load_mnist
from tinynn.gradient import numerical_gradient
from tinynn.functions import (sigmoid, softmax, cross_entropy_error,
                                   sigmoid_grad)
import numpy as np


class SimpleTwoLayerNet:
    """ 最简单的两层神经网络，包括前向和反向 """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 weight_init_std: float = 0.01) -> None:
        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(
            input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(
            hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x: np.ndarray) -> np.ndarray:
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        return softmax(a2)

    def loss(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x: np.ndarray, t: np.ndarray):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        return np.sum(y == t) / float(x.shape[0])

    def numerical_gradient(self, x: np.ndarray, t: np.ndarray) -> dict:
        """ 计算梯度方式1：普通方式计算梯度 """

        func = lambda _: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(func, self.params['W1'])
        grads['b1'] = numerical_gradient(func, self.params['b1'])
        grads['W2'] = numerical_gradient(func, self.params['W2'])
        grads['b2'] = numerical_gradient(func, self.params['b2'])
        return grads

    def gradient(self, x: np.ndarray, t: np.ndarray) -> dict:
        """ 计算梯度方式2：使用反向传播方式高速计算梯度 """
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads


def test() -> None:
    net = SimpleTwoLayerNet(input_size=784, hidden_size=100, output_size=10)
    print(net.params['W1'].shape)  # (784, 100)
    x = np.random.rand(100, 784)  # 伪输入数据(100笔)
    t = np.random.rand(100, 10)  # 伪正确解标签(100笔)
    y = net.predict(x)
    print(y.shape)
    # grads = net.numerical_gradient(x, t)
    grads = net.gradient(x, t)
    print(grads)


def train() -> None:
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True,
                                                      one_hot_label=True)
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    # 超参数
    iters_num = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1
    iter_per_epoch = max(train_size / batch_size, 1)

    network = SimpleTwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    for i in range(iters_num):
        # 获取mini-batch
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        # 计算梯度
        # grad = network.numerical_gradient(x_batch, t_batch)
        grad = network.gradient(x_batch, t_batch)  # 高速版!(反向传播)
        # 更新参数
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]
        # 记录学习过程
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)
        # 计算每个epoch的识别精度
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print(f"train acc, test acc | {train_acc}, {test_acc}")


if __name__ == '__main__':
    # train()
    test()
