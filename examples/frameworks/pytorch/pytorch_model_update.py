from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from ignite.contrib.handlers import TensorboardLogger
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import global_step_from_engine
from ignite.metrics import Accuracy, Loss, Recall
from ignite.utils import setup_logger
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from clearml import Task, StorageManager, OutputModel


def main():
    # Connecting ClearML with the current process,
    # from here on everything is logged automatically
    task = Task.init(
        project_name="examples",
        task_name="Model update PyTorch",
        auto_connect_frameworks={"pytorch": False},
    )
    params = {
        "number_of_epochs": 1,
        "batch_size": 64,
        "dropout": 0.25,
        "base_lr": 0.001,
        "momentum": 0.9,
        "loss_report": 100,
    }

    params = task.connect(params)  # enabling configuration override by clearml
    print(params)  # printing actual configuration (after override in remote mode)

    model = OutputModel(task=task, framework="pytorch")
    model_config_dict = {
        "list_of_ints": [1, 2, 3, 4],
        "dict": {
            "sub_value": "string",
            "sub_integer": 11
            },
        "value": 13.37
    }
    model.update_design(config_dict=model_config_dict)

    manager = StorageManager()

    dataset_path = Path(
        manager.get_local_copy(
            remote_url="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        )
    )

    # Dataset and Dataloader initializations
    transform = transforms.Compose([transforms.ToTensor()])

    trainset = datasets.CIFAR10(
        root=dataset_path, train=True, download=False, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=params.get("batch_size", 4), shuffle=True, num_workers=10
    )

    testset = datasets.CIFAR10(
        root=dataset_path, train=False, download=False, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=params.get("batch_size", 4), shuffle=False, num_workers=10
    )

    run(
        params["number_of_epochs"],
        params["base_lr"],
        params["momentum"],
        10,
        params,
        trainloader,
        testloader,
        model,
    )


# Helper function to store predictions and scores using matplotlib
def predictions_gt_images_handler(engine, logger, *args, **kwargs):
    x, _ = engine.state.batch
    y_pred, y = engine.state.output

    num_x = num_y = 4
    le = num_x * num_y
    fig = plt.figure(figsize=(20, 20))
    trans = transforms.ToPILImage()
    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )
    enumeration = {k: v for v, k in enumerate(classes, 1)}
    Task.current_task().connect_label_enumeration(enumeration)

    for idx in range(le):
        preds = torch.argmax(F.softmax(y_pred[idx], dim=0))
        probs = torch.max(F.softmax(y_pred[idx], dim=0))
        ax = fig.add_subplot(num_x, num_y, idx + 1, xticks=[], yticks=[])
        ax.imshow(trans(x[idx]))
        ax.set_title(
            "{0} {1:.1f}% (label: {2})".format(
                classes[preds], probs * 100, classes[y[idx]]
            ),
            color=("green" if preds == y[idx] else "red"),
        )
    logger.writer.add_figure(
        "predictions vs actuals", figure=fig, global_step=engine.state.epoch
    )


class Net(nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.dorpout = nn.Dropout(p=params.get("dropout", 0.25))
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(self.dorpout(x))
        return x


# Training
def run(epochs, lr, momentum, log_interval, params, trainloader, testloader, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = Net(params).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

    trainer = create_supervised_trainer(net, optimizer, criterion, device=device)
    trainer.logger = setup_logger("trainer")

    val_metrics = {"accuracy": Accuracy(), "loss": Loss(criterion), "recall": Recall()}
    evaluator = create_supervised_evaluator(net, metrics=val_metrics, device=device)
    evaluator.logger = setup_logger("evaluator")

    # Attach handler to plot trainer's loss every 100 iterations
    tb_logger = TensorboardLogger(log_dir="cifar-output")
    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED(every=params.get("loss_report")),
        tag="training",
        output_transform=lambda loss: {"loss": loss},
    )

    # Attach handler to dump evaluator's metrics every epoch completed
    for tag, evaluator in [("training", trainer), ("validation", evaluator)]:
        tb_logger.attach_output_handler(
            evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag=tag,
            metric_names="all",
            global_step_transform=global_step_from_engine(trainer),
        )

    # Attach function to build debug images and report every epoch end
    tb_logger.attach(
        evaluator,
        log_handler=predictions_gt_images_handler,
        event_name=Events.EPOCH_COMPLETED(once=1),
    )

    desc = "ITERATION - loss: {:.2f}"
    pbar = tqdm(initial=0, leave=False, total=len(trainloader), desc=desc.format(0))

    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_training_loss(engine):
        pbar.desc = desc.format(engine.state.output)
        pbar.update(log_interval)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        pbar.refresh()
        evaluator.run(trainloader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics["accuracy"]
        avg_nll = metrics["loss"]
        tqdm.write(
            "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}".format(
                engine.state.epoch, avg_accuracy, avg_nll
            )
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(testloader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics["accuracy"]
        avg_nll = metrics["loss"]
        tqdm.write(
            "Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}".format(
                engine.state.epoch, avg_accuracy, avg_nll
            )
        )

        pbar.n = pbar.last_print_n = 0

    @trainer.on(Events.EPOCH_COMPLETED | Events.COMPLETED)
    def log_time():
        tqdm.write(
            "{} took {} seconds".format(
                trainer.last_event_name.name,
                trainer.state.times[trainer.last_event_name.name],
            )
        )

    trainer.run(trainloader, max_epochs=epochs)
    pbar.close()

    PATH = "./cifar_net.pth"

    # CONDITION depicts a custom condition for when to save the model. The model is saved and then updated in ClearML
    CONDITION = True

    if CONDITION:
        torch.save(net.state_dict(), PATH)
        model.update_weights(weights_filename=PATH)
    print("Finished Training")
    print("Task ID number is: {}".format(Task.current_task().id))


if __name__ == "__main__":
    main()
