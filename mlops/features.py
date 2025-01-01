from torch.utils.data import DataLoader


def get_dataloaders(train, test, batch_size=2 ** 4):
    train_dataloader = DataLoader(train, batch_size, shuffle=True)
    test_dataloader = DataLoader(test, batch_size, shuffle=False)

    return train_dataloader, test_dataloader
