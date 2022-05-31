# ClearML - Fastai v2 with tensorboard callbacks example code, automatic logging the model and various scalars
#
import argparse

from clearml import Task

import fastai

try:
    from fastai.vision.all import (
        untar_data,
        URLs,
        get_image_files,
        ImageDataLoaders,
        Resize,
        vision_learner,
        resnet34,
        error_rate,
    )
    from fastai.callback.tensorboard import TensorBoardCallback
except ImportError:
    raise ImportError("FastAI version %s imported, but this example is for FastAI v2." % fastai.__version__)


def label_func(f):
    return f[0].isupper()


def main(epochs):
    Task.init(project_name="examples", task_name="fastai v2 with tensorboard callback")

    path = untar_data(URLs.PETS)
    files = get_image_files(path / "images")

    dls = ImageDataLoaders.from_name_func(path, files, label_func, item_tfms=Resize(224), num_workers=0)
    dls.show_batch()
    learn = vision_learner(dls, resnet34, metrics=error_rate)
    learn.fine_tune(epochs, cbs=[TensorBoardCallback()])
    learn.show_results()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=3)
    args = parser.parse_args()
    main(args.epochs)
