from typing import Optional
import numpy as np

from tinynn.optimizer import (SGD, AdaGrad, Adam, Momentum, RMSprop)


class Trainer:
    """ 辅助进行神经网络的训练的类 """

    def __init__(self,
                 network: object,
                 x_train: np.ndarray,
                 t_train: np.ndarray,
                 x_test: np.ndarray,
                 t_test: np.ndarray,
                 epochs: int = 20,
                 mini_batch_size: int = 100,
                 optimizer: str = 'SGD',
                 optimizer_param: dict = {'lr': 0.01},
                 evaluate_sample_num_per_epoch: Optional[int] = None,
                 verbose: bool = True) -> None:
        self.network = network
        self.verbose = verbose
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch

        # optimzer
        optimizer_class_dict = {
            'sgd': SGD,
            'momentum': Momentum,
            'adagrad': AdaGrad,
            'rmsprpo': RMSprop,
            'adam': Adam,
        }
        # 传入对应字符串，小写化后选择对应的优化器类
        # 随后把optimizer_param作参数传进去创建优化器类
        self.optimizer = optimizer_class_dict[optimizer.lower()](
            **optimizer_param)

        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1)
        self.max_iter = int(epochs * self.iter_per_epoch)
        self.current_iter = 0
        self.current_epoch = 0

        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def train_step(self) -> None:
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]

        grads = self.network.gradient(x_batch, t_batch)
        self.optimizer.update(self.network.params, grads)

        loss = self.network.loss(x_batch, t_batch)
        self.train_loss_list.append(loss)
        if self.verbose:
            print(f"train loss: {loss}")

        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1

            x_train_sample, t_train_sample = self.x_train, self.t_train
            x_test_sample, t_test_sample = self.x_test, self.t_test
            if not self.evaluate_sample_num_per_epoch is None:
                t = self.evaluate_sample_num_per_epoch
                x_train_sample, t_train_sample = self.x_train[:
                                                              t], self.t_train[:
                                                                               t]
                x_test_sample, t_test_sample = self.x_test[:t], self.t_test[:t]

            train_acc = self.network.accuracy(x_train_sample, t_train_sample)
            test_acc = self.network.accuracy(x_test_sample, t_test_sample)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            if self.verbose:
                print(
                    f"=== epoch: {self.current_epoch}, train acc: {train_acc}, "
                    f"test acc: {test_acc} ===")
        self.current_iter += 1

    def train(self) -> None:
        for _ in range(self.max_iter):
            self.train_step()

        test_acc = self.network.accuracy(self.x_test, self.t_test)

        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print(f"test acc: {test_acc}")
