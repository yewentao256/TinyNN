import numpy as np
import matplotlib.pyplot as plt
from tinynn.full_connect_network import MultiLayerNet
from tinynn.mnist import load_mnist
from tinynn.trainer import Trainer

if __name__ == '__main__':

    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

    # 为了再现过拟合，减少学习数据
    x_train = x_train[:300]
    t_train = t_train[:300]

    # 设定是否使用Dropuout，以及比例 ========================
    use_dropout = True  # 不使用Dropout的情况下为False
    dropout_ratio = 0.2
    # ====================================================

    network = MultiLayerNet(input_size=784,
                            hidden_size_list=[100, 100, 100, 100, 100, 100],
                            output_size=10,
                            use_dropout=use_dropout,
                            dropout_ration=dropout_ratio)
    # 封装好的训练者类，可传递网络和数据集进去，直接进行训练。
    trainer = Trainer(network,
                      x_train,
                      t_train,
                      x_test,
                      t_test,
                      epochs=301,
                      mini_batch_size=100,
                      optimizer='sgd',
                      optimizer_param={'lr': 0.01},
                      verbose=True)
    trainer.train()

    # 绘制图形==========
    markers = {'train': 'o', 'test': 's'}
    x = np.arange(len(trainer.train_acc_list))
    plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=10)
    plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=10)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.show()