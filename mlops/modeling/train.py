import os
import time

from torch.nn.functional import *
import numpy as np
import torch


def xavier_normal(F_in, F_out):
    limit = np.sqrt(6 / float(F_in + F_out))
    W = np.random.uniform(low=-limit, high=limit, size=(F_in, F_out))
    return torch.from_numpy(W).type(torch.float32).requires_grad_()


class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate, *args, **kwargs):
        # Инициализация параметров
        super().__init__(*args, **kwargs)
        self.w1 = torch.nn.Parameter(xavier_normal(hidden_size, input_size))
        self.b1 = torch.nn.Parameter(torch.randn(hidden_size, requires_grad=True))

        self.w2 = torch.nn.Parameter(xavier_normal(output_size, hidden_size))
        self.b2 = torch.nn.Parameter(torch.randn(output_size, requires_grad=True))

        self.lr = learning_rate

    # Функция прямого прохода
    def forward(self, x):
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

    def export_to_onnx(self, input_example):
        export_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..',
            '..',
            'models', f'{int(time.time())}.onnx')
        torch.onnx.export(
            self,
            input_example,  # Пример входных данных
            export_path,  # Путь сохранения
            export_params=True,  # Сохранение параметров
            opset_version=11,  # Версия ONNX
            do_constant_folding=True,  # Оптимизация постоянных выражений
            input_names=['input'],  # Имена входных тензоров
            output_names=['output'],  # Имена выходных тензоров
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
