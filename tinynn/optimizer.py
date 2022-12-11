import numpy as np


class Optimizer(object):

    def update(self, params: dict, grads: dict) -> None:
        self._update(params, grads)

    def _update(self, *_) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """ 随机梯度下降法(Stochastic Gradient Descent) """

    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def _update(self, params: dict, grads: dict) -> None:
        # python中字典属于传引用
        for key in params.keys():
            # W = W - lr * grads
            params[key] -= self.lr * grads[key]


class Momentum(Optimizer):
    """Momentum SGD"""

    def __init__(self, lr: float = 0.01, momentum: float = 0.9) -> None:
        self.lr = lr
        self.momentum = momentum
        self.velocity: dict = None

    def _update(self, params: dict, grads: dict) -> None:
        if self.velocity is None:
            # 初始化velocity字典
            self.velocity = {}
            for key, val in params.items():
                self.velocity[key] = np.zeros_like(val)

        for key in params.keys():
            # v = momentum*v_old - lr * grad
            self.velocity[key] = self.momentum * self.velocity[
                key] - self.lr * grads[key]
            params[key] += self.velocity[key]


class AdaGrad(Optimizer):
    """AdaGrad，学习率自动衰减的优化器"""

    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr
        self.h = None

    def _update(self, params: dict, grads: dict) -> None:
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            # h += grad ^ 2
            # W -= lr * grad / sqrt(h)
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class RMSprop(Optimizer):
    """RMSprop
    
    一般AdaGrad的平方和不断累加，就会导致越到后面更新量越小，趋于0。

    RMSProp方法并不是将过去所有的梯度一视同仁地相加，
    而是逐渐地遗忘过去的梯度，在做加法运算时将新梯度的信息更多地反映出来
    """

    def __init__(self, lr: float = 0.01, decay_rate: float = 0.99):
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None

    def _update(self, params: dict, grads: dict) -> None:
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            # h * decay_rate 让h不会无限增加
            self.h[key] *= self.decay_rate
            self.h[key] += (1 - self.decay_rate) * grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class Adam(Optimizer):
    """Adam (http://arxiv.org/abs/1412.6980v8)
    
    Momentum + Adagrad 的思路合并形成Adam
    """

    def __init__(self,
                 lr: float = 0.001,
                 beta1: float = 0.9,
                 beta2: float = 0.999):
        self.lr = lr
        self.beta1 = beta1  # 第一次momentum系数
        self.beta2 = beta2  # 第二次momentum系数
        self.iter = 0
        self.m: dict = None
        self.v: dict = None

    def _update(self, params: dict, grads: dict) -> None:
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1
        # 新学习率 = lr * sqrt(1 - beta2*iter) / (1 - beta1 ** iter)
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (
            1.0 - self.beta1**self.iter)

        for key in params.keys():
            # self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            # self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])

            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)

            # unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
            # unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
            # params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)
