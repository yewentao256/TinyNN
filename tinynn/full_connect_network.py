from typing import Union
import numpy as np
from collections import OrderedDict
from tinynn.gradient import numerical_gradient
from tinynn.layers import Affine, BatchNormalization, Dropout, Relu, Sigmoid, SoftmaxWithLoss


class MultiLayerNet:
    """全连接的多层神经网络
    
    具有Weiht Decay、Dropout、Batch Normalization的功能
    """

    def __init__(self,
                 input_size: int,
                 hidden_size_list: list,
                 output_size: int,
                 activation: str = 'relu',
                 weight_init_std: Union[list, str] = 'relu',
                 weight_decay_lambda: float = 0,
                 use_dropout: bool = False,
                 dropout_ration: int = 0.5,
                 use_batch_normalization: bool = False) -> None:
        """
        Args:
            input_size (int): 输入大小(MNIST的情况下为784)
            hidden_size_list (list): 隐藏层的神经元数量的列表
                (e.g. [100, 100, 100])
            output_size (int): 输出大小(MNIST的情况下为10)
            activation (str, optional): 'relu' or 'sigmoid'. Defaults to 'relu'.
            weight_init_std (str, optional): 指定权重的标准差(e.g. 0.01)
                指定'relu'或'he'的情况下设定“He的初始值”
                指定'sigmoid'或'xavier'的情况下设定“Xavier的初始值”.
                Defaults to 'relu'.
            weight_decay_lambda (float, optional): Weight Decay(L2范数)的强度.
                Defaults to 0.
            use_dropout (bool, optional): 是否使用Dropout. Defaults to False.
            dropout_ration (int, optional): dropout比例. Defaults to 0.5.
            use_batch_normalization (bool, optional): 是否使用Batch Normalization. 
                Defaults to False.
        """
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.use_dropout = use_dropout
        self.weight_decay_lambda = weight_decay_lambda
        self.use_batch_normalization = use_batch_normalization
        self.params = {}

        # 初始化权重
        self.__init_weight(weight_init_std)

        # 生成层
        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu}
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num + 1):
            self.layers['Affine' + str(idx)] = Affine(
                self.params['W' + str(idx)], self.params['b' + str(idx)])
            if self.use_batch_normalization:
                self.params['gamma' + str(idx)] = np.ones(
                    hidden_size_list[idx - 1])
                self.params['beta' + str(idx)] = np.zeros(
                    hidden_size_list[idx - 1])
                self.layers['BatchNorm' + str(idx)] = BatchNormalization(
                    self.params['gamma' + str(idx)],
                    self.params['beta' + str(idx)])

            self.layers['Activation_function' +
                        str(idx)] = activation_layer[activation]()

            if self.use_dropout:
                self.layers['Dropout' + str(idx)] = Dropout(dropout_ration)

        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                  self.params['b' + str(idx)])

        self.last_layer = SoftmaxWithLoss()

    def __init_weight(self, weight_init_std: Union[str, float]) -> None:
        """设定权重的初始值

        Args:
            weight_init_std (Union[str, float]): 指定权重的标准差(e.g. 0.01)
                指定'relu'或'he'的情况下设定“He的初始值”
                指定'sigmoid'或'xavier'的情况下设定“Xavier的初始值”
        """

        all_size_list = [self.input_size
                         ] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 /
                                all_size_list[idx - 1])  # 使用ReLU的情况下推荐的初始值
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 /
                                all_size_list[idx - 1])  # 使用sigmoid的情况下推荐的初始值
            self.params['W' + str(idx)] = scale * np.random.randn(
                all_size_list[idx - 1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

    def predict(self, x: np.ndarray, train_flg: bool = False) -> np.ndarray:
        for key, layer in self.layers.items():
            if "Dropout" in key or "BatchNorm" in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)

        return x

    def loss(self,
             x: np.ndarray,
             t: np.ndarray,
             train_flg: bool = False) -> np.ndarray:
        """求损失函数
        参数x是输入数据，t是教师标签
        """
        y = self.predict(x, train_flg)

        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2)

        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x: np.ndarray, t: np.ndarray) -> float:
        y = self.predict(x, train_flg=False)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        return np.sum(y == t) / float(x.shape[0])

    def numerical_gradient(self, x: np.ndarray, t: np.ndarray) -> dict:
        """ 求梯度(数值微分) """
        loss_W = lambda W: self.loss(x, t, train_flg=True)

        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            grads['W' + str(idx)] = numerical_gradient(
                loss_W, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(
                loss_W, self.params['b' + str(idx)])

            if self.use_batch_normalization and idx != self.hidden_layer_num + 1:
                grads['gamma' + str(idx)] = numerical_gradient(
                    loss_W, self.params['gamma' + str(idx)])
                grads['beta' + str(idx)] = numerical_gradient(
                    loss_W, self.params['beta' + str(idx)])

        return grads

    def gradient(self, x: np.ndarray, t: np.ndarray) -> dict:
        # forward
        self.loss(x, t, train_flg=True)

        # backward
        dout = self.last_layer.backward(1)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            grads['W' + str(idx)] = self.layers['Affine' + str(
                idx)].dW + self.weight_decay_lambda * self.params['W' +
                                                                  str(idx)]
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

            if self.use_batch_normalization and idx != self.hidden_layer_num + 1:
                grads['gamma' + str(idx)] = self.layers['BatchNorm' +
                                                        str(idx)].dgamma
                grads['beta' + str(idx)] = self.layers['BatchNorm' +
                                                       str(idx)].dbeta

        return grads