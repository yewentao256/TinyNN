from tinynn.mnist import load_mnist

if __name__ == '__main__':
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True,
                                                      normalize=False,
                                                      one_hot_label=True)
    img = x_train[0]
    label = t_train[0]
    print(label)  # 5
    print(img.shape)  # (784,)
    img = img.reshape(28, 28)
    print(img.shape)  # (28, 28)
