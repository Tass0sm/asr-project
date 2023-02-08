import torch

import argparse


from model.model_factory import get_model
from optimizer.optimizer_factory import get_optimizer
from loss.loss_factory import get_loss


print("torch cuda is_available {}".format(torch.cuda.is_available()))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("-e", "--epoch", type=int, default=300, metavar='>= 0', help="Number of epoch")

parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="Learning rate")
parser.add_argument("-b", "--batch_size", type=int, default=16, metavar='>= 0', help="Batch size")
parser.add_argument("-oc", "--output_class", type=int, default=50, metavar='>= 0', help="Number of class at the output")
parser.add_argument("-mn", "--model_name", type=str, default="AxialAttentionWithoutPosition",
                    choices=["AxialAttentionWithoutPosition",
                             "AxialAttentionPosition",
                             "AxialAttentionPositionGate"],
                    help="Name of model")
parser.add_argument("-on", "--optimizer_name", type=str, default="Adam",
                    choices=["Adam", "SGD", "RMSprop", "AdamW",  "SparseAdam"],
                    help="Optimizer name")
parser.add_argument("-ln", "--loss_name", type=str, default="CrossEntropy",
                    choices=["CrossEntropy"],
                    help="Loss function name")
args = parser.parse_args()

model = get_model(args.model_name, args.output_class)
model = model.to(device)
optimizer = get_optimizer(args.optimizer_name, args.learning_rate, model)
loss_fn = get_loss(args.loss_name)


for epoch_index in range(args.epoch):
    running_loss = 0.
    last_loss = 0.
    for i, data in enumerate(training_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            # tb_x = epoch_index * len(training_loader) + i + 1
            # tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.
