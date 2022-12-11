from typing import Optional
import numpy as np
from tinynn.functions import cross_entropy_error, sigmoid, softmax
from tinynn.util import im2col, col2im


class Layer(object):
    """ Layer基类 """

    def forward(self, x: np.ndarray, *args: list[object]) -> np.ndarray:
        return self._forward(x, *args) if args else self._forward(x)
    
    def _forward(self, *_) -> np.ndarray:
        raise NotImplementedError

    def backward(self, dout: np.ndarray, *args: list[object]) -> np.ndarray:
        return self._backward(dout, *args) if args else self._backward(dout)

    def _backward(self, *_) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"Layer: {self.__class__.__name__}"

class Relu(Layer):

    def __init__(self) -> None:
        # mask是由True/False构成的NumPy数组，
        # 它会把正向传播时的输入x的元素中小于等于0的地方保存为True，
        # 其他地方(大于0的元素)保存为False。
        self.mask = None

    def _forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def _backward(self, dout: np.ndarray) -> np.ndarray:
        # RELU层，x>=0的导数不变，x<=0的导数为0
        dout[self.mask] = 0
        dx = dout
        return dx


class Sigmoid(Layer):

    def __init__(self) -> None:
        self.out = None

    def _forward(self, x: np.ndarray) -> np.ndarray:
        self.out = sigmoid(x)
        return self.out

    def _backward(self, dout: np.ndarray) -> np.ndarray:
        # y = 1 / (1 + exp(-x))
        # 画出反向传播图，得到dx = dout * y^2 * e^(-x)，简化后得到下式
        return dout * (1.0 - self.out) * self.out


class Affine(Layer):

    def __init__(self, W: np.ndarray, b: np.ndarray) -> None:
        self.W = W
        self.b = b

        self.x = None
        self.original_x_shape = None
        # 权重和偏置参数的导数
        self.dW = None
        self.db = None

    def _forward(self, x: np.ndarray) -> np.ndarray:
        # y = x * W + b
        self.original_x_shape = x.shape
        # 这样操作可以支持四维张量，统一转化为二维矩阵后再计算
        self.x = x.reshape(x.shape[0], -1)
        return np.dot(self.x, self.W) + self.b

    def _backward(self, dout: np.ndarray) -> np.ndarray:
        # dx = dout * W的转置(涉及矩阵求导)
        dx = np.dot(dout, self.W.T)
        # dW = x的转置 * dout(涉及矩阵求导)
        self.dW = np.dot(self.x.T, dout)
        # db = dout第零轴方向上的和(涉及矩阵求导)
        self.db = np.sum(dout, axis=0)

        return dx.reshape(*self.original_x_shape)  # 还原形状


class SoftmaxWithLoss(Layer):
    # Softmax + Cross Entropy Error

    def __init__(self) -> None:
        self.loss: np.ndarray = None
        self.y: np.ndarray = None  # softmax的输出
        self.t: np.ndarray = None  # 监督数据

    def _forward(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def _backward(self, dout=1) -> np.ndarray:

        batch_size = self.t.shape[0]

        # 对应标签t是one-hot矩阵的情况
        # 注：一定要除以batch_size，因为偏置有一个求和的过程，使用np.sum
        # 会将同一轴数据加和，如batch_size为2
        # 求得y-t = [[0.3,-0.8,0.5],[-0.7,0.2,0.5]]
        # y-t恰好是softmax with loss反向传播的结果，有兴趣的同学可以推导一下
        # sum后变为[-0.4,-0.6,1]显然不是单个数据求出的正常值
        # 除以batch_size才会得到单个数据的正常结果[-0.2,-0.3,0.5]
        if self.t.size == self.y.size:
            dx = (self.y - self.t) * dout
            dx = dx / batch_size

        else:
            dx = self.y.copy() * dout
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        return dx


class Dropout(Layer):
    """
    dropout层，可参考http://arxiv.org/abs/1207.0580
    """

    def __init__(self, dropout_ratio: int = 0.5) -> None:
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def _forward(self, x: np.ndarray, train_flg: bool = True) -> np.ndarray:
        # train_flg用于标记是测试还是训练阶段
        if train_flg:
            # np.random.rand可以返回一个或一组服从“0~1”均匀分布的随机样本值。
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            # 如果非train，最终应乘上训练时删除掉的比例(如删掉10%，则最后乘以0.9)
            # 以实现整体的“dropout”
            return x * (1.0 - self.dropout_ratio)

    def _backward(self, dout: np.ndarray) -> np.ndarray:
        # 反向传播时，遇到mask数组中false的(删除的)，则返回0信号(或说不返回信号)
        return dout * self.mask


class BatchNormalization(Layer):
    """
    批标准化：以进行学习时的mini-batch为单位，按mini batch进行标准化
    
    具体而言，就是进行使数据分布的均值为0、方差为1的标准化
    均值 u_mean = sum(x)/len(x)
    方差 var = sum((x - u_mean)^2) / len(x)
    最终 x_new = (x - u_mean) / sqrt(var + 微小值) —— 减均值除标准差 

    此外，会将x做一个平移和缩放，一开始gamma为1，beta为0
    即 y = gamma * x_new + beta

    详见 http://arxiv.org/abs/1502.03167
    """

    def __init__(self,
                 gamma: np.ndarray,
                 beta: np.ndarray,
                 momentum: int = 0.9,
                 running_mean: Optional[float] = None,
                 running_var: Optional[float] = None) -> None:
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None

        # 测试时使用的平均值和方差
        self.running_mean = running_mean
        self.running_var = running_var

        # backward时使用的中间数据
        self.batch_size = None
        self.x_sub_mean_u = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def _forward(self, x: np.ndarray, train_flg: bool = True) -> np.ndarray:
        self.input_shape = x.shape
        if x.ndim != 2:
            # 高维矩阵全部转为二维
            # 本项目中，Conv层的情况下为4维，全连接层的情况下为2维
            x = x.reshape(x.shape[0], -1)

        if self.running_mean is None:
            _, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        # 训练时
        if train_flg:
            # 计算均值方差，然后减均值除标准差
            mean_u = x.mean(axis=0)
            x_sub_mean_u = x - mean_u
            var = np.mean(x_sub_mean_u**2, axis=0)
            std = np.sqrt(var + 10e-7)
            x_new = x_sub_mean_u / std

            self.batch_size = x.shape[0]
            self.x_sub_mean_u = x_sub_mean_u
            self.x_new = x_new
            self.std = std
            # 按momentum更新运行时均值和标准差
            self.running_mean = self.momentum * self.running_mean + (
                1 - self.momentum) * mean_u
            self.running_var = self.momentum * self.running_var + (
                1 - self.momentum) * var

        # 测试时
        else:
            x_sub_mean_u = x - self.running_mean
            x_new = x_sub_mean_u / ((np.sqrt(self.running_var + 10e-7)))

        out = self.gamma * x_new + self.beta

        return out.reshape(*self.input_shape)

    def _backward(self, dout: np.ndarray) -> np.ndarray:
        if dout.ndim != 2:
            dout = dout.reshape(dout.shape[0], -1)

        # BN的反向传播有些复杂，参考博客
        # Understanding the backward pass through Batch Normalization Layer
        self.dgamma = dout.sum(axis=0)
        self.dbeta = np.sum(self.x_new * dout, axis=0)

        dx_new = self.gamma * dout
        dx_sub_mean_u = dx_new / self.std
        dstd = -np.sum(
            (dx_new * self.x_sub_mean_u) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dx_sub_mean_u += (2.0 / self.batch_size) * self.x_sub_mean_u * dvar
        dmean_u = np.sum(dx_sub_mean_u, axis=0)

        dx = dx_sub_mean_u - dmean_u / self.batch_size

        return dx.reshape(*self.input_shape)


class Convolution(Layer):
    """ 卷积层 """

    def __init__(self,
                 W: np.ndarray,
                 b: np.ndarray,
                 stride: int = 1,
                 pad: int = 0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        # 中间数据(backward时使用)
        self.x = None
        self.col = None
        self.col_W = None

        # 权重和偏置参数的梯度
        self.dW = None
        self.db = None

    def _forward(self, x: np.ndarray) -> np.ndarray:
        # 卷积的前向运算，本质就是 y = W * X + b
        # 其中X是输入数组，一般是NCHW，W是卷积核，一般是FCHW（F为卷积核个数）
        self.x = x
        FN, _, FH, FW = self.W.shape
        N, _, H, W = x.shape

        # im2col后得到了卷积核扫过二维矩阵，两个展开矩阵运算恰好能完成卷积过程
        self.col = im2col(x, FH, FW, self.stride, self.pad)  # 使用im2col展开图像
        self.col_W = self.W.reshape(FN, -1).T  # 滤波器的展开，-1会自动计算个数

        # 计算输出形状
        out_h = 1 + int((H + 2 * self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2 * self.pad - FW) / self.stride)

        out: np.ndarray = np.dot(self.col, self.col_W) + self.b
        # transpose转换轴的顺序，  (N,H,W,C) → (N,C,H,W)
        return out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

    def _backward(self, dout: np.ndarray) -> np.ndarray:
        # 卷积的反向传播类似affine的逻辑，分别求出db、dW、dx即可

        FN, C, FH, FW = self.W.shape
        # transpose成NHWC，reshape为二维矩阵
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        # db = dout第零轴方向上的和(涉及矩阵求导)
        self.db = np.sum(dout, axis=0)

        # dW = x的转置 * dout(涉及矩阵求导)
        # 但这里注意还需要将dW的形状转回去
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        # dx = dout * W的转置(涉及矩阵求导)
        # 同样注意将形状转回去，col2im二维转四维
        dcol = np.dot(dout, self.col_W.T)
        return col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)


class Pooling(Layer):
    """ Max pooling """

    def __init__(self,
                 pool_h: int,
                 pool_w: int,
                 stride: int = 1,
                 pad: int = 0) -> None:
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None  # 最大值位置，用于反向传播

    def _forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x

        N, C, H, W = x.shape
        # 计算输出形状
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        # 二维展开image
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        # reshape成每行为一层池化扫描过的矩阵
        col = col.reshape(-1, self.pool_h * self.pool_w)  # N * 扫描个数

        # 取最大值索引并寻找最大值
        self.arg_max = np.argmax(col, axis=1)
        # out: np.ndarray = col[range(col.shape[0]), self.arg_max]
        # 相当于out = np.max(col, axis=1)，但这样直接求值相当于算两次最大值
        # 这里取值方法是迭代 range => data[0, arg[0]], data[1, arg[1]]...
        # 也是官方文档中推荐的做法，详情见
        # https://numpy.org/doc/stable/user/quickstart.html#indexing-with-arrays-of-indices
        out: np.ndarray = col[range(col.shape[0]), self.arg_max]

        # 知识：NCHW中，按照[W H C N]的顺序放元素，先走W再走H最后是C和N
        # 以RGB为例即 'RRRRRR GGGGGG BBBBBB' 这种形式
        # 而NHWC按照[C W H N] 的顺序放元素，先走C再走W最后是H和N
        # 以RGB为例即 'RGB RGB RGB RGB RGB RGB' 这种形式

        # 这里我们需要NCHW的输出，但如果直接reshape成NCHW的话
        # 输出的并不是按卷积核扫过的顺序（有疑惑的同学可以断点调试看一下输出）
        # 所以我们借助NHWC中间态然后reshape成NCHW
        return out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

    def _backward(self, dout: np.ndarray) -> np.ndarray:
        """ Pooling层最大池化的反向传播
        
        1、padding
        2、映射回原最大值位置，其他取0
        """
        # reshape成NHWC
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w

        # 临时二维矩阵dmax，N * 滤波核面积，例如432000 * 4
        dmax = np.zeros((dout.size, pool_size))

        # 将dout中，对应之前arg_max的位置赋上导数值，其他都为0
        dmax[np.arange(self.arg_max.size),
             self.arg_max.flatten()] = dout.flatten()

        # reshape后，变为五维，NHWC + size，如(100, 12, 12, 30, 4)
        dmax = dmax.reshape(dout.shape + (pool_size, ))

        # 再使用dmax reshape成二维dcol的形状， 如(14400, 120)
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)

        # 调用col2im返回最终反向结果
        # 为什么要转来转去呢？就可以不用或者少用低效率的for循环来处理
        return col2im(dcol, self.x.shape, self.pool_h, self.pool_w,
                      self.stride, self.pad)
