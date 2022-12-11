import numpy as np
import matplotlib.pyplot as plt
from tinynn.mnist import load_mnist
from tinynn.full_connect_network import MultiLayerNet
from tinynn.util import shuffle_dataset
from tinynn.trainer import Trainer


def train(lr: float,
          weight_decay: float,
          epochs: int = 50) -> tuple[list, list]:
    network = MultiLayerNet(input_size=784,
                            hidden_size_list=[100, 100, 100, 100, 100, 100],
                            output_size=10,
                            weight_decay_lambda=weight_decay)
    # optimizer_param指选择对应优化器时应传进去的参数
    trainer = Trainer(network,
                      x_train,
                      t_train,
                      x_val,
                      t_val,
                      epochs=epochs,
                      mini_batch_size=100,
                      optimizer='sgd',
                      optimizer_param={'lr': lr},
                      verbose=False)
    trainer.train()
    return trainer.test_acc_list, trainer.train_acc_list


if __name__ == '__main__':
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

    # 为了实现高速化，减少训练数据
    x_train = x_train[:1000]
    t_train = t_train[:1000]

    # 分割验证数据
    validation_rate = 0.20
    #注意要加int，否则默认为float是无法作为index的
    validation_num = int(x_train.shape[0] * validation_rate)
    x_train, t_train = shuffle_dataset(x_train, t_train)  #打乱数据集

    x_val = x_train[:validation_num]
    t_val = t_train[:validation_num]
    x_train = x_train[validation_num:]
    t_train = t_train[validation_num:]

    # ============超参数的随机搜索=================
    # 通过设定一个搜索范围，逐个实验，找到最优参数
    optimization_trial = 100
    results_val = {}
    results_train = {}
    for _ in range(optimization_trial):
        # 指定搜索的超参数的范围===============
        # 我们这里搜索权重衰减值以及学习率
        weight_decay = 10**np.random.uniform(-8, -4)  #10的负8次方到10的负4次方
        lr = 10**np.random.uniform(-6, -2)
        # ================================================

        val_acc_list, train_acc_list = train(lr, weight_decay)
        print(f"val acc: {val_acc_list[-1]} | lr: {lr}"
              f", weight decay: {weight_decay}")
        #把随机出来的学习率和权重衰减(带值)一整个作为key
        key = f"lr: {lr}, weight decay: {weight_decay}"
        results_val[key] = val_acc_list
        results_train[key] = train_acc_list

    # 绘制图形========================================================
    print("=========== Hyper-Parameter Optimization Result ===========")
    graph_draw_num = 20
    col_num = 5
    # np.ceil向上取整，如-0.2取整为0.0
    row_num = int(np.ceil(graph_draw_num / col_num))
    i = 0
    # key=lambda x:x[1][-1]即对results_val中的第二维数据(即values),中的最后一个值
    # reverse=true表示由高到低进行排序
    for key, val_acc_list in sorted(results_val.items(),
                                    key=lambda x: x[1][-1],
                                    reverse=True):
        print(f"Best-{i+1} (val acc: {val_acc_list[-1]}) | {key}")

        plt.subplot(row_num, col_num, i + 1)
        plt.title(f"Best-{i+1}")
        plt.ylim(0.0, 1.0)
        #注意：i%5，i=1结果是1，i=0结果是0
        if i % 5:
            plt.yticks([])  #每五幅图创建一个y轴(当i%5结果为0是不删除y轴)
        plt.xticks([])  #(删除所有x轴)
        x = np.arange(len(val_acc_list))
        plt.plot(x, val_acc_list)  #图中实线表示验证集精度
        plt.plot(x, results_train[key], "--")  #图中虚线表示训练集精度
        i += 1

        if i >= graph_draw_num:
            break

    #最后筛选出能正常进行学习的超参数，再进一步筛选
    plt.show()
