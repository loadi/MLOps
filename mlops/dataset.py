# Загрузка обучающей и тестовой выборки
from torchvision import datasets
from torchvision.transforms import ToTensor
import os

def get_dataset():
    dir = os.path.join(os.getcwd(), '..', 'data')
    os.makedirs(dir, exist_ok=True)

    train = os.path.join(dir, 'train')
    test = os.path.join(dir, 'test')

    train_set = datasets.EMNIST(
        root=train,
        train=True,
        download=True,
        split='letters',
        transform=ToTensor()
    )

    test_set = datasets.EMNIST(
        root=test,
        train=False,
        download=True,
        split='letters',
        transform=ToTensor()
    )
    return train_set, test_set