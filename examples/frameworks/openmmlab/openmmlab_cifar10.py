import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from mmcv.parallel import MMDataParallel
from mmcv.runner import EpochBasedRunner
from mmcv.utils import get_logger
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train_step(self, data, optimizer):
        images, labels = data
        predicts = self(images)  # -> self.__call__() -> self.forward()
        loss = self.loss_fn(predicts, labels)
        return {'loss': loss}


if __name__ == '__main__':
    model = Model()
    if torch.cuda.is_available():
        model = MMDataParallel(model.cuda(), device_ids=[0])

    # dataset and dataloader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = CIFAR10(
        root='data', train=True, download=True, transform=transform)
    trainloader = DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    logger = get_logger('mmcv')
    # runner is a scheduler to manage the training
    runner = EpochBasedRunner(
        model,
        optimizer=optimizer,
        work_dir='./work_dir',
        logger=logger,
        max_epochs=4)

    # learning rate scheduler config
    lr_config = dict(policy='step', step=[2, 3])
    # configuration of optimizer
    optimizer_config = dict(grad_clip=None)
    # configuration of saving checkpoints periodically
    checkpoint_config = dict(interval=1)
    # save log periodically and multiple hooks can be used simultaneously
    log_config = dict(
        interval=100,
        hooks=[
            dict(type='TextLoggerHook'),
            dict(
                type='ClearMLLoggerHook',
                init_kwargs=dict(
                    project_name='examples',
                    task_name='OpenMMLab cifar10',
                    output_uri=True
                )
            ),
        ]
    )
    # register hooks to runner and those hooks will be invoked automatically
    runner.register_training_hooks(
        lr_config=lr_config,
        optimizer_config=optimizer_config,
        checkpoint_config=checkpoint_config,
        log_config=log_config)

    runner.run([trainloader], [('train', 1)])
