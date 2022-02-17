import argparse
import os
import sys
from tempfile import gettempdir

import numpy as np

try:
    import megengine as mge
    import megengine.functional as F
    import megengine.module as M
    from megengine.autodiff import GradManager
    from megengine.data import DataLoader, RandomSampler
    from megengine.data.dataset import MNIST
    from megengine.data.transform import Compose, Normalize, Pad, ToMode
    from megengine.optimizer import SGD
except ImportError:
    raise ImportError(
        "megengine package is missing, you can install it using pip: pip install megengine"
        if sys.version_info.minor <= 8
        else "MegEngine does not support python version >= 3.9"
    )

from clearml import Task
from tensorboardX import SummaryWriter


class Net(M.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = M.Conv2d(1, 20, kernel_size=5, bias=False)
        self.bn0 = M.BatchNorm2d(20)
        self.relu0 = M.ReLU()
        self.pool0 = M.MaxPool2d(2)
        self.conv1 = M.Conv2d(20, 20, kernel_size=5, bias=False)
        self.bn1 = M.BatchNorm2d(20)
        self.relu1 = M.ReLU()
        self.pool1 = M.MaxPool2d(2)
        self.fc0 = M.Linear(500, 64, bias=True)
        self.relu2 = M.ReLU()
        self.fc1 = M.Linear(64, 10, bias=True)

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.pool0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = F.flatten(x, 1)
        x = self.fc0(x)
        x = self.relu2(x)
        x = self.fc1(x)
        return x


def build_dataloader():
    train_dataset = MNIST(root=gettempdir(), train=True, download=True)
    dataloader = DataLoader(
        train_dataset,
        transform=Compose([Normalize(mean=0.1307 * 255, std=0.3081 * 255), Pad(2), ToMode("CHW"),]),
        sampler=RandomSampler(dataset=train_dataset, batch_size=64),
    )
    return dataloader


def train(dataloader, args):
    writer = SummaryWriter("runs")
    net = Net()
    net.train()
    optimizer = SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    gm = GradManager().attach(net.parameters())

    epoch_length = len(dataloader)
    for epoch in range(args.epoch):
        for step, (batch_data, batch_label) in enumerate(dataloader):
            batch_label = batch_label.astype(np.int32)
            data, label = mge.tensor(batch_data), mge.tensor(batch_label)
            with gm:
                pred = net(data)
                loss = F.loss.cross_entropy(pred, label)
                gm.backward(loss)
            optimizer.step().clear_grad()

            if step % 50 == 0:
                print("epoch:{}, iter:{}, loss:{}".format(epoch + 1, step, float(loss)))  # noqa
            writer.add_scalar("loss", float(loss), epoch * epoch_length + step)
        if (epoch + 1) % 5 == 0:
            mge.save(
                net.state_dict(), os.path.join(gettempdir(), f"mnist_net_e{epoch + 1}.pkl"),
            )  # noqa


def main():
    task = Task.init(project_name="examples", task_name="MegEngine MNIST train")  # noqa

    parser = argparse.ArgumentParser(description="MegEngine MNIST Example")
    parser.add_argument(
        "--epoch", type=int, default=10, help="number of training epoch(default: 10)",
    )
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate(default: 0.01)")
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        "--wd", type=float, default=5e-4, help="SGD weight decay(default: 5e-4)",
    )

    args = parser.parse_args()
    dataloader = build_dataloader()
    train(dataloader, args)


if __name__ == "__main__":
    main()
