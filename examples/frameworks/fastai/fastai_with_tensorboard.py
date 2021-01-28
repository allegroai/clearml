# ClearML - Fastai with Tensorboard example code, automatic logging the model and Tensorboard outputs
#

from fastai.callbacks.tensorboard import LearnerTensorboardWriter
from fastai.vision import *  # Quick access to computer vision functionality

from clearml import Task

# Connecting ClearML with the current process,
# from here on everything is logged automatically
task = Task.init(project_name="example", task_name="fastai with tensorboard callback")

path = untar_data(URLs.MNIST_SAMPLE)

data = ImageDataBunch.from_folder(path, ds_tfms=(rand_pad(2, 28), []), bs=64)
data.normalize(imagenet_stats)

learn = cnn_learner(data, models.resnet18, metrics=accuracy)
tboard_path = Path("data/tensorboard/project1")
learn.callback_fns.append(
    partial(LearnerTensorboardWriter, base_dir=tboard_path, name="run0")
)

accuracy(*learn.get_preds())
learn.fit_one_cycle(6, 0.01)
