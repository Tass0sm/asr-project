import torch.nn as nn


def get_loss(loss_name):
    if loss_name == "CrossEntropy":
        return nn.CrossEntropyLoss()


