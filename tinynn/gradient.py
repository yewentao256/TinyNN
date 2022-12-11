# coding: utf-8
import numpy as np


def numerical_gradient(function: object, darray: np.ndarray) -> np.ndarray:
    """对某个矩阵使用某函数来求梯度

    Args:
        function (object): 求微分函数
        darray (np.ndarray): 需处理的矩阵

    Returns:
        np.ndarray: 梯度
    """
    
    minor_value = 1e-4  # 0.0001
    grad = np.zeros_like(darray)

    # np.nditer用于迭代数组
    it = np.nditer(darray, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = darray[idx]
        darray[idx] = float(tmp_val) + minor_value
        fxh1 = function(darray)  # f(x+h)

        darray[idx] = tmp_val - minor_value
        fxh2 = function(darray)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*minor_value)

        darray[idx] = tmp_val  # 还原值
        it.iternext()

    return grad
