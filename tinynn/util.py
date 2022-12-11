import numpy as np


def smooth_curve(x: np.ndarray) -> np.ndarray:
    """用于使损失函数的图形变圆滑

    参考：http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
    """
    window_len = 11
    s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[5:len(y) - 5]


def shuffle_dataset(x: np.ndarray,
                    t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ 打乱数据集 """
    permutation = np.random.permutation(x.shape[0])
    x = x[permutation, :] if x.ndim == 2 else x[permutation, :, :, :]
    t = t[permutation]
    return x, t


def im2col(input_data: np.ndarray,
           filter_h: int,
           filter_w: int,
           stride: int = 1,
           pad: int = 0) -> np.ndarray:
    """将图像展开为二维矩阵，展开后可以减少几层for循环的卷积运算

    其本质为将一次滤波器应用的区域的数据横向展开为一列

    Args:
        input_data (np.ndarray): 由(数据量, 通道, 高, 长)的4维数组构成的输入数据
            即NCHW
        filter_h (int): 滤波器的高
        filter_w (int): 滤波器的长
        stride (int, optional): 步幅. Defaults to 1.
        pad (int, optional): 填充. Defaults to 0.

    Examples

        x1 = np.random.rand(1, 3, 7, 7)  # 批大小1，通道3，7*7数据
        col1 = im2col(x1, 5, 5, stride=1, pad=0)   # 滤波器通道3，大小5*5
        print(col1.shape) # (9, 75)

        x2 = np.random.rand(10, 3, 7, 7) # 10个数据
        col2 = im2col(x2, 5, 5, stride=1, pad=0)
        print(col2.shape) # (90, 75)

    Returns:
        np.ndarray: 2维数组

    """
    N, C, H, W = input_data.shape

    # 计算卷积处理后的形状公式
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    # np.pad(constant)对input data的第三四维填充p
    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)],
                 'constant')
    # 六维数组col (N, C, filter_h, filter_w, out_h, out_w)
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for j in range(filter_h):
        j_max = j + stride * out_h
        for i in range(filter_w):
            i_max = i + stride * out_w
            # 知识：np.array[1:5:2]指的是从index1拿到index4(右开)的数据，stride为2
            # 这里即将img中j~j_max(H), i~i_max(W)的数据，以stride步长赋给col
            # j~j_max(H), i~i_max(W) 即一次卷积核处理扫过的所有区域
            col[:, :, j, i, :, :] = img[:, :, j:j_max:stride, i:i_max:stride]

    # transpose后，col变为 (N, out_h, out_W, C, filter_h, filter_w)
    # 这样调整后再reshape为 (N * out_h * out_w)行, C * filter_h * filter_w列的数据
    # 即每一列都为一次滤波器应用后的区域
    return col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)


def col2im(col: np.ndarray,
           input_shape: tuple,
           filter_h: int,
           filter_w: int,
           stride: int = 1,
           pad: int = 0) -> np.ndarray:
    """将二维矩阵展开为NCHW

    im2col方法的逆过程。注意：只适用于dout的输入，即backward的时候使用

    Args:
        col (np.ndarray): im2col得到的二维矩阵
        input_shape (tuple): 输入形状
        filter_h (int): 滤波器高
        filter_w (int): 滤波器长
        stride (int, optional): 步长. Defaults to 1.
        pad (int, optional): 填充量. Defaults to 0.

    Returns:
        np.ndarray: NCHW
    """
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    # 六维数组col (N, filter_h, filter_w, C, out_h, out_w)
    col = col.reshape(N, out_h, out_w, C, filter_h,
                      filter_w).transpose(0, 3, 4, 5, 1, 2)

    # 目标img NCHW
    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))

    for j in range(filter_h):
        j_max = j + stride * out_h
        for i in range(filter_w):
            i_max = i + stride * out_w
            # 这里即将col中j~j_max(H), i~i_max(W)的数据，以stride步长赋给img
            # j~j_max(H), i~i_max(W) 即一次卷积核处理扫过的所有区域
            # 注意此处部分区域可能会多次加和，所以只适用于backward阶段
            img[:, :, j:j_max:stride, i:i_max:stride] += col[:, :, j, i, :, :]

    # pad:H+pad 而不是 0:H，不会将pad的值算入img
    return img[:, :, pad:H + pad, pad:W + pad]