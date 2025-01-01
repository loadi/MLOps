from torch.nn.functional import *
import numpy as np
import torch


def xavier_normal(F_in, F_out):
    limit = np.sqrt(6 / float(F_in + F_out))
    W = np.random.uniform(low=-limit, high=limit, size=(F_in, F_out))
    return torch.from_numpy(W).type(torch.float32).requires_grad_()


class Model:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        # Инициализация параметров
        self.w1 = xavier_normal(hidden_size, input_size)
        self.b1 = torch.randn(hidden_size, requires_grad=True)

        self.w2 = xavier_normal(output_size, hidden_size)
        self.b2 = torch.randn(output_size, requires_grad=True)

        self.lr = learning_rate

    # Функция прямого прохода
    def predict(self, x):
        x = x.flatten(start_dim=1)  # преобразование тензора в плоский
        z1 = linear(x, self.w1, self.b1)  # линейное преобразование
        a1 = relu(z1)  # ф. актив. ReLU
        z2 = linear(a1, self.w2, self.b2)
        y_pred = log_softmax(z2, dim=1)

        return y_pred

    # Обратное распространение
    def back_propagate(self, y_pred, y_expected):
        loss = nll_loss(y_pred, y_expected)
        loss.backward()

        # Обновление параметров
        with torch.no_grad():
            self.w1 -= self.lr * self.w1.grad
            self.b1 -= self.lr * self.b1.grad
            self.w2 -= self.lr * self.w2.grad
            self.b2 -= self.lr * self.b2.grad

            # Сброс градиентов
            self.w1.grad.zero_()
            self.b1.grad.zero_()
            self.w2.grad.zero_()
            self.b2.grad.zero_()
        return loss
