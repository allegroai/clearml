# Example of MNIST training with PyTorch and abseil integration

from __future__ import print_function

import os
from tempfile import gettempdir

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from absl import app, flags
from torchvision import datasets, transforms

from clearml import Task, Logger

# Training settings
FLAGS = flags.FLAGS

flags.DEFINE_integer("batch_size", 64, help="Batch size for training")
flags.DEFINE_integer("test_batch_size", 1000, help="Batch size for testing")
flags.DEFINE_integer("epochs", 10, help="Number of epochs to train")
flags.DEFINE_float("lr", 0.01, help="Learning rate")
flags.DEFINE_float("momentum", 0.5, help="SGD momentum")
flags.DEFINE_bool("cuda", True, help="Enable CUDA training")
flags.DEFINE_integer("seed", 1, help="Random seed")
flags.DEFINE_integer(
    "log_interval", 10, help="How many batches to wait before logging training status"
)
flags.DEFINE_bool("save_model", True, help="For saving the current Model")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            Logger.current_logger().report_scalar(
                "train",
                "loss",
                iteration=(epoch * len(train_loader) + batch_idx),
                value=loss.item(),
            )
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test(args, model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    Logger.current_logger().report_scalar(
        "test", "loss", iteration=epoch, value=test_loss
    )
    Logger.current_logger().report_scalar(
        "test", "accuracy", iteration=epoch, value=(correct / len(test_loader.dataset))
    )
    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def main(_):
    # Connecting ClearML with the current process,
    # from here on everything is logged automatically
    task = Task.init(project_name="examples", task_name="PyTorch MNIST train with abseil")

    use_cuda = FLAGS.cuda and torch.cuda.is_available()

    torch.manual_seed(FLAGS.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            os.path.join("..", "data"),
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=FLAGS.batch_size,
        shuffle=True,
        **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            os.path.join("..", "data"),
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=FLAGS.test_batch_size,
        shuffle=True,
        **kwargs
    )

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=FLAGS.lr, momentum=FLAGS.momentum)

    for epoch in range(1, FLAGS.epochs + 1):
        train(FLAGS, model, device, train_loader, optimizer, epoch)
        test(FLAGS, model, device, test_loader, epoch)

    if FLAGS.save_model:
        torch.save(model.state_dict(), os.path.join(gettempdir(), "mnist_cnn_abseil.pt"))


if __name__ == "__main__":
    app.run(main)
