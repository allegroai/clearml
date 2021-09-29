# ClearML - Example of pytorch with tensorboardX add_video.
#
from __future__ import print_function

import torch
from tensorboardX import SummaryWriter

from clearml import Task


def main():

    # Connecting ClearML with the current process,
    # from here on everything is logged automatically
    task = Task.init(project_name="examples", task_name="pytorch with video tensorboardX")

    writer = SummaryWriter("runs")
    writer.add_text("TEXT", "This is some text", 0)

    # Make a video that simply fades grey colors
    video = (torch.sin(torch.arange(0, 1000) / 100) + 1) / 2 * 255
    video = video.byte().view(1, -1, 1, 1, 1).expand(1, -1, 3, 64, 64)

    writer.add_video("my_video", video, 0, fps=50)


if __name__ == "__main__":
    main()
