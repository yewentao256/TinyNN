class MulLayer:
    """简单的乘法层，支持float的前向和反向"""

    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x: float, y: float) -> float:
        self.x = x
        self.y = y
        out = x * y
        return out

    def backward(self, dout: float) -> float:
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy


class AddLayer:
    """简单的加法层，支持float的前向和反向"""

    def forward(self, x: float, y: float) -> float:
        return x + y

    def backward(self, dout: float) -> float:
        # z = x + y
        dx = dout * 1
        dy = dout * 1
        return dx, dy


if __name__ == '__main__':

    apple = 100
    apple_num = 2
    orange = 150
    orange_num = 3
    tax = 1.1
    # layer
    mul_apple_layer = MulLayer()
    mul_orange_layer = MulLayer()
    add_apple_orange_layer = AddLayer()
    mul_tax_layer = MulLayer()
    # forward
    apple_price = mul_apple_layer.forward(apple, apple_num)  # (1)
    orange_price = mul_orange_layer.forward(orange, orange_num)  # (2)
    all_price = add_apple_orange_layer.forward(apple_price,
                                               orange_price)  # (3)
    price = mul_tax_layer.forward(all_price, tax)  # (4)
    # backward
    dprice = 1
    dall_price, dtax = mul_tax_layer.backward(dprice)  # (4)
    dapple_price, dorange_price = add_apple_orange_layer.backward(
        dall_price)  # (3)
    dorange, dorange_num = mul_orange_layer.backward(dorange_price)  # (2)
    dapple, dapple_num = mul_apple_layer.backward(dapple_price)  # (1)
    print(price)  # 715
    print(dapple_num, dapple, dorange, dorange_num,
          dtax)  # 110 2.2 3.3 165 650
