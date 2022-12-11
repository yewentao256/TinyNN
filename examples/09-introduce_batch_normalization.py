import numpy as np
import matplotlib.pyplot as plt
from tinynn.full_connect_network import MultiLayerNet
from tinynn.mnist import load_mnist
from tinynn.optimizer import SGD


def train(weight_init_std: float) -> tuple[list, list]:
    bn_network = MultiLayerNet(
        input_size=784,
        hidden_size_list=[100, 100, 100, 100, 100],
        output_size=10,
        weight_init_std=weight_init_std,
        # use_batch_normalization=False
        use_batch_normalization=True)
    network = MultiLayerNet(input_size=784,
                            hidden_size_list=[100, 100, 100, 100, 100],
                            output_size=10,
                            weight_init_std=weight_init_std)
    optimizer = SGD(lr=learning_rate)

    train_acc_list = []
    bn_train_acc_list = []

    iter_per_epoch = max(train_size / batch_size, 1)
    epoch_cnt = 0

    for i in range(1000000000):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        for _network in (bn_network, network):
            grads = _network.gradient(x_batch, t_batch)
            optimizer.update(_network.params, grads)

        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            bn_train_acc = bn_network.accuracy(x_train, t_train)
            train_acc_list.append(train_acc)
            bn_train_acc_list.append(bn_train_acc)

            print(f"epoch: {epoch_cnt} | {train_acc} - {bn_train_acc}")

            epoch_cnt += 1
            if epoch_cnt >= max_epochs:
                break

    return train_acc_list, bn_train_acc_list


if __name__ == '__main__':
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

    # 减少学习数据
    x_train = x_train[:1000]
    t_train = t_train[:1000]

    max_epochs = 10
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.01

    # 3.绘制图形==========
    weight_scale_list = np.logspace(0, -4,
                                    num=16)  #创建基数为10的，16个数据的从1到10的-4次方的的等比数列
    x = np.arange(max_epochs)

    for i, w in enumerate(weight_scale_list):
        print(f"============== {i + 1} / 16 ==============")
        train_acc_list, bn_train_acc_list = train(w)

        plt.subplot(4, 4, i + 1)
        plt.title(f"W: {w}")
        if i == 15:
            plt.plot(x,
                     bn_train_acc_list,
                     label='Batch Normalization',
                     markevery=2)
            plt.plot(x,
                     train_acc_list,
                     linestyle="--",
                     label='Normal(without BatchNorm)',
                     markevery=2)
        else:
            plt.plot(x, bn_train_acc_list, markevery=2)
            plt.plot(x, train_acc_list, linestyle="--", markevery=2)

        plt.ylim(0, 1.0)
        if i % 4:
            #最终会生成4排16个图形，这样做每四个只会生成一个y轴
            plt.yticks([])
        else:
            plt.ylabel("accuracy")
        if i < 12:
            #前12个图形不生成x轴
            plt.xticks([])
        else:
            plt.xlabel("epochs")
        plt.legend(loc='lower right')

    plt.show()