import urllib.request
import os.path
import gzip
import pickle
import os
import numpy as np


def _load_label(file_path: str) -> np.ndarray:

    print(f"Converting {file_path} to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done")

    return labels


def _load_img(file_path: str) -> np.ndarray:
    print(f"Converting {file_path} to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, 784)
    print("Done")

    return data


def init_mnist(dataset_dir: str) -> None:
    url_base = 'http://yann.lecun.com/exdb/mnist/'
    key_file = {
        'train_img': 'train-images-idx3-ubyte.gz',
        'train_label': 'train-labels-idx1-ubyte.gz',
        'test_img': 't10k-images-idx3-ubyte.gz',
        'test_label': 't10k-labels-idx1-ubyte.gz'
    }
    for file_name in key_file.values():
        file_path = dataset_dir + file_name

        if not os.path.exists(file_path):
            print(f"Downloading {file_name}... ")
            urllib.request.urlretrieve(url_base + file_name, file_path)
            print("Done")

    dataset = {}
    dataset['train_img'] = _load_img(dataset_dir + key_file['train_img'])
    dataset['train_label'] = _load_label(dataset_dir + key_file['train_label'])
    dataset['test_img'] = _load_img(dataset_dir + key_file['test_img'])
    dataset['test_label'] = _load_label(dataset_dir + key_file['test_label'])

    print("Creating pickle file ...")
    with open(dataset_dir + "mnist.pkl", 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done!")


def _change_one_hot_label(labels: np.ndarray) -> np.ndarray:
    one_hot_labels = np.zeros((labels.size, 10))
    for idx, row in enumerate(one_hot_labels):
        row[labels[idx]] = 1

    return one_hot_labels


def load_mnist(
    normalize: bool = True,
    flatten: bool = True,
    one_hot_label: bool = False
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """读入MNIST数据集，如果没有则下载到 {workdir}/dataset目录下

    Args:
        normalize (bool, optional): 将图像的像素值正规化为0.0~1.0.
            Defaults to True.
        flatten (bool, optional): 是否将图像展开为一维数组. Defaults to True.
        one_hot_label (bool, optional): 标签是否作为one-hot数组返回.
            Defaults to False.

    Returns:
        Tuple[tuple, tuple]: (训练图像, 训练标签), (测试图像, 测试标签)
    """
    dataset_dir = f'{os.getcwd()}/dataset/'
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
    if not os.path.exists(dataset_dir + "mnist.pkl"):
        init_mnist(dataset_dir)

    with open(dataset_dir + "mnist.pkl", 'rb') as f:
        dataset = pickle.load(f)

    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32) / 255.0

    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])

    if not flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset['train_img'],
            dataset['train_label']), (dataset['test_img'],
                                      dataset['test_label'])
