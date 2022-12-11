from tinynn.gradient import numerical_gradient
from tinynn.functions import softmax, cross_entropy_error
import numpy as np


class simpleNet:

    def __init__(self):
        self.W = np.random.randn(2, 3)  # 用标准正态分布进行初始化，二行三列的数组。

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss


if __name__ == '__main__':

    net = simpleNet()
    print(f'初始化权重矩阵: {net.W}')
    x = np.array([0.6, 0.9])
    print(f'初始化输入值: {x}')
    p = net.predict(x)
    print(f'forward结果: {p}')
    print(f'最大值索引: {np.argmax(p)}')
    t = np.array([0, 0, 1])
    print(f'loss: {net.loss(x, t)}')

    func = lambda _: net.loss(x, t)  # 定义一个无参的匿名函数，调用net.loss来求导

    dW = numerical_gradient(func, net.W)  # 对权重基于损失函数求导
    print(f'权重矩阵梯度: {dW}')  # 正值表示应该参数应向负方向更新，来减少损失函数。同理负值的对应参数应该向正方向更新
