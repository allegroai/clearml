# ClearML - example of ClearML torch distributed support
# notice all nodes will be reporting to the master Task (experiment)

import os
import subprocess
import sys
from argparse import ArgumentParser
from math import ceil
from random import Random

import torch as th
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms

from clearml import Task


local_dataset_path = './MNIST_data'


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = th.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class Partition(object):
    """ Dataset partitioning helper """
    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    def __init__(self, data, sizes=(0.7, 0.2, 0.1), seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


def partition_dataset(num_workers=4):
    """ Partitioning MNIST """
    dataset = datasets.MNIST(root=local_dataset_path, train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))
                             ]))
    size = dist.get_world_size()
    bsz = int(128 / float(size))
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = th.utils.data.DataLoader(
        partition, num_workers=num_workers, batch_size=bsz, shuffle=True)
    return train_set, bsz


def run(num_workers):
    """ Distributed Synchronous SGD Example """
    th.manual_seed(1234)
    train_set, bsz = partition_dataset(num_workers)
    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    num_batches = ceil(len(train_set.dataset) / float(bsz))

    from random import randint
    param = {'worker_{}_stuff'.format(dist.get_rank()): 'some stuff ' + str(randint(0, 100))}
    Task.current_task().connect(param)
    Task.current_task().upload_artifact(
        'temp {:02d}'.format(dist.get_rank()), artifact_object={'worker_rank': dist.get_rank()})

    for epoch in range(2):
        epoch_loss = 0.0
        for i, (data, target) in enumerate(train_set):
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss.item()
            loss.backward()
            average_gradients(model)
            optimizer.step()
            if i % 10 == 0:
                print('{}] Train Epoch {} - {} \tLoss  {:.6f}'.format(dist.get_rank(), epoch, i, loss))
                Task.current_task().get_logger().report_scalar(
                    'loss', 'worker {:02d}'.format(dist.get_rank()), value=loss.item(), iteration=i)
            if i > 100:
                break
        print('Rank ', dist.get_rank(), ', epoch ',
              epoch, ': ', epoch_loss / num_batches)


def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--nodes', help='number of nodes', type=int, default=10)
    parser.add_argument('--workers_in_node', help='number of workers per node', type=int, default=3)
    # this argument we will not be logging, see below Task.init
    parser.add_argument('--rank', help='current rank', type=int)

    args = parser.parse_args()

    # We have to initialize the task in the master process,
    # it will make sure that any sub-process calling Task.init will get the master task object
    # notice that we exclude the `rank` argument, so we can launch multiple sub-processes with clearml-agent
    # otherwise, the `rank` will always be set to the original value.
    task = Task.init("examples", "test torch distributed", auto_connect_arg_parser={'rank': False})

    if not dist.is_available():
        print("torch.distributed is not supported for this platform")
        exit(0)

    if os.environ.get('MASTER_ADDR'):
        dist.init_process_group(backend='gloo', rank=args.rank, world_size=args.nodes)
        run(args.workers_in_node)
    else:
        # first let's download the dataset, if we have multiple machines,
        # they will take care of it when they get there
        datasets.MNIST(root=local_dataset_path, train=True, download=True)

        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'

        print(os.getpid(), 'ARGS:', args)
        processes = []
        for rank in range(args.nodes):
            cmd = [sys.executable, sys.argv[0],
                   '--nodes', str(args.nodes),
                   '--workers_in_node', str(args.workers_in_node),
                   '--rank', str(rank)]
            print(cmd)
            p = subprocess.Popen(cmd, cwd=os.getcwd(), pass_fds=[], close_fds=True)
            processes.append(p)

        for p in processes:
            p.wait()
