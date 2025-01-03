import torch


def get_model_accuracy(model, test_dataloader):
    correct = 0

    for batch_idx, (x, y) in enumerate(test_dataloader):
        y_pred = model.forward(x)
        correct += (y_pred.argmax(1) == y).type(torch.float).sum().item()
    return correct / len(test_dataloader.dataset)