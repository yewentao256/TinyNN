import numpy as np
from tinynn.gradient import numerical_gradient
from tinynn.mnist import load_mnist
from collections import OrderedDict

from tinynn.layers import Affine, Layer, Relu, SoftmaxWithLoss


class TwoLayerNet:

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
        # 生成层
        self.layers: OrderedDict[str, Layer] = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x: np.ndarray, t: np.ndarray) -> float:
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if (t.ndim != 1):
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x: np.ndarray, t: np.ndarray) -> dict:
        loss_W = lambda _: self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads

    def gradient(self, x: np.ndarray, t: np.ndarray) -> dict:
        # forward
        self.loss(x, t)
        # backward
        dout = self.lastLayer.backward(1)
        layers = list(self.layers.values())  # 将字典里保存的所有层化为列表
        layers.reverse()  # 反序以反向传播
        for layer in layers:
            dout = layer.backward(dout)
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        return grads


def test() -> None:
    # 检测误差反向传播法与数值微分是否相同。如果差距较大，则反向传播法存在问题
    # 读入数据
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True,
                                                      one_hot_label=True)
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    x_batch = x_train[:3]
    t_batch = t_train[:3]
    grad_numerical = network.numerical_gradient(x_batch, t_batch)
    grad_backprop = network.gradient(x_batch, t_batch)
    # 求各个权重的绝对误差的平均值
    for key in grad_numerical.keys():
        diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
        print(f"{key}: {diff}")


def train() -> None:
    # 读入数据
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True,
                                                      one_hot_label=True)
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    iters_num = 10000
    train_size = x_train.shape[0]  #60000
    batch_size = 100
    learning_rate = 0.1
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    iter_per_epoch = max(train_size / batch_size, 1)  #每一个epoch输出信息
    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        # 通过误差反向传播法求梯度
        grad = network.gradient(x_batch, t_batch)
        # 更新
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print(f"train acc, test acc | {train_acc}, {test_acc}")


if __name__ == '__main__':
    train()
    # test()