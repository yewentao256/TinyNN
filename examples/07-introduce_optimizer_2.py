import matplotlib.pyplot as plt
from tinynn.optimizer import SGD, AdaGrad, Adam, Momentum, Optimizer, RMSprop
from tinynn.mnist import load_mnist
from tinynn.util import smooth_curve
from tinynn.full_connect_network import MultiLayerNet
import numpy as np

if __name__ == '__main__':
    # 0:读入MNIST数据==========
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

    train_size = x_train.shape[0]
    batch_size = 128
    max_iterations = 201

    # 1:进行实验的设置==========
    optimizers: dict[str, Optimizer] = {}  #用字典存放优化器
    optimizers['SGD'] = SGD()
    optimizers['Momentum'] = Momentum()
    optimizers['AdaGrad'] = AdaGrad()
    optimizers['Adam'] = Adam()
    optimizers['RMSprop'] = RMSprop()

    networks: dict[str, MultiLayerNet] = {}
    train_loss = {}
    for key in optimizers.keys():
        # 每个优化器建立一个神经网络
        networks[key] = MultiLayerNet(input_size=784,
                                      hidden_size_list=[100, 100, 100, 100],
                                      output_size=10)
        train_loss[key] = []

    # 2:开始训练==========
    for i in range(max_iterations):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        for key in optimizers.keys():
            grads = networks[key].gradient(x_batch, t_batch)
            optimizers[key].update(networks[key].params, grads)

            loss = networks[key].loss(x_batch, t_batch)
            train_loss[key].append(loss)

        if i % 100 == 0:
            print(f"=========== iteration: {i} ===========")
            for key in optimizers.keys():
                loss = networks[key].loss(x_batch, t_batch)
                print(f"{key}: {loss}")

    # 3.绘制图形==========
    markers = {
        "SGD": "o",
        "Momentum": "x",
        "AdaGrad": "s",
        "Adam": "D",
        "RMSprop": "o"
    }
    x = np.arange(max_iterations)
    for key in optimizers.keys():
        # 注：soomth_curve用于使损失函数图形变得圆滑
        # marker是标记，100轮标记一次
        # o是圆形，x是叉号，s是正方向，d是菱形
        plt.plot(
            x,
            smooth_curve(train_loss[key]),
            marker=markers[key],
            markevery=100,
            label=key)
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.ylim(0, 1)  #设定y轴为0~1
    plt.legend()  #给图加上图例
    plt.show()
