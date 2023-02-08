from torch import optim


def get_optimizer(optimizer_name, lr, model):
    if optimizer_name == "Adam":
        return optim.Adam(model.parameters(), lr=lr)
    if optimizer_name == "AdamW":
        return optim.AdamW(model.parameters(), lr=lr)
    if optimizer_name == "SparseAdam":
        return optim.SparseAdam(model.parameters(), lr=lr)
    if optimizer_name == "SGD":
        return optim.SGD(model.parameters(), lr=lr)
    if optimizer_name == "RMSprop":
        return optim.RMSprop(model.parameters(), lr=lr)

