import numpy as np


def sigmoid(x: np.ndarray) -> float:
    return 1 / (1 + np.exp(-x))    


def sigmoid_grad(x: np.ndarray) -> float:
    # 计算sigmoid的梯度
    return (1.0 - sigmoid(x)) * sigmoid(x)
    

def softmax(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy_error(y: np.ndarray, t: np.ndarray) -> float:
    """ 交叉熵误差 """

    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
