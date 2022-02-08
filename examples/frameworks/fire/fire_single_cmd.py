# ClearML - Example of Python Fire integration, with a single command passed to Fire
#
from clearml import Task

import fire


def hello(count, name="clearml", prefix="prefix_", suffix="_suffix", **kwargs):
    for _ in range(count):
        print("Hello %s%s%s!" % (prefix, name, suffix))


if __name__ == "__main__":
    Task.init(project_name="examples", task_name="Fire single command")
    fire.Fire(hello)
